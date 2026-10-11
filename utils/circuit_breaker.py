"""Redis circuit breaker with owned, generation-fenced recovery probes."""
import os
import uuid
from dataclasses import dataclass
import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True, socket_connect_timeout=1, socket_timeout=1,
    retry=Retry(NoBackoff(), 0),
)

class ServiceUnavailable(Exception):
    pass

@dataclass(frozen=True)
class Permit:
    token: str
    generation: int
    probe: bool

_ADMIT = """
redis.replicate_commands()
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local state = redis.call('HGET', KEYS[1], 'state') or 'closed'
local generation = tonumber(redis.call('HGET', KEYS[1], 'generation') or '0')
if state == 'closed' then return {generation, 0} end
if state == 'open' and now < tonumber(redis.call('HGET', KEYS[1], 'retry_at') or '0') then return {} end
if not redis.call('SET', KEYS[2], ARGV[1], 'NX', 'EX', ARGV[2]) then return {} end
generation = redis.call('HINCRBY', KEYS[1], 'generation', 1)
redis.call('HSET', KEYS[1], 'state', 'half-open')
return {generation, 1}
"""
_FINISH = """
redis.replicate_commands()
local generation = tonumber(redis.call('HGET', KEYS[1], 'generation') or '0')
if generation ~= tonumber(ARGV[2]) then return 0 end
local state = redis.call('HGET', KEYS[1], 'state') or 'closed'
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
if ARGV[3] == '1' then
    if state ~= 'half-open' or redis.call('GET', KEYS[2]) ~= ARGV[1] then return 0 end
    redis.call('DEL', KEYS[2])
    redis.call('HINCRBY', KEYS[1], 'generation', 1)
    if ARGV[4] == 'success' then
        redis.call('HMSET', KEYS[1], 'state', 'closed', 'failures', 0, 'retry_at', 0)
    else
        redis.call('HMSET', KEYS[1], 'state', 'open', 'retry_at', now + tonumber(ARGV[6]))
    end
elseif state == 'closed' then
    if ARGV[4] == 'success' then redis.call('HSET', KEYS[1], 'failures', 0)
    elseif ARGV[4] == 'failure' then
        if redis.call('HINCRBY', KEYS[1], 'failures', 1) >= tonumber(ARGV[5]) then
            redis.call('HINCRBY', KEYS[1], 'generation', 1)
            redis.call('HMSET', KEYS[1], 'state', 'open', 'retry_at', now + tonumber(ARGV[6]))
        end
    end
else return 0 end
return 1
"""

class CircuitBreaker:
    def __init__(self, service_name, failure_threshold=3, cooldown=60,
                 *, connection=None, probe_seconds=90):
        self.redis = connection if connection is not None else redis_client
        namespace = os.getenv("SERVICE_CONTROL_NAMESPACE", "studywise")
        self.key = f"{namespace}:control:{{{service_name}}}:breaker:v2"
        self.probe_key = self.key + ":probe"
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.probe_seconds = probe_seconds

    def state(self):
        return self.redis.hget(self.key, "state") or "closed"

    def is_open(self):
        # Observational only: metrics must never claim the recovery probe.
        return self.state() != "closed"

    def acquire(self):
        token = uuid.uuid4().hex
        try:
            result = self.redis.eval(_ADMIT, 2, self.key, self.probe_key, token, self.probe_seconds)
        except redis.RedisError as exc:
            raise ServiceUnavailable("Service admission unavailable") from exc
        if not result:
            raise ServiceUnavailable("Service circuit is open or recovering")
        return Permit(token, int(result[0]), bool(result[1]))

    def validate(self, permit):
        return bool(self.redis.eval("""
            local generation = tonumber(redis.call('HGET', KEYS[1], 'generation') or '0')
            if generation ~= tonumber(ARGV[2]) then return 0 end
            local state = redis.call('HGET', KEYS[1], 'state') or 'closed'
            if ARGV[3] == '1' then
                if state == 'half-open' and redis.call('GET', KEYS[2]) == ARGV[1] then return 1 end
            elseif state == 'closed' then return 1 end
            return 0
        """, 2, self.key, self.probe_key, permit.token, permit.generation, int(permit.probe)))

    def renew_probe(self, permit):
        return bool(self.redis.eval("""
            if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
            return redis.call('EXPIRE', KEYS[1], ARGV[2])
        """, 1, self.probe_key, permit.token, self.probe_seconds))

    def finish(self, permit, outcome):
        return bool(self.redis.eval(_FINISH, 2, self.key, self.probe_key,
            permit.token, permit.generation, int(permit.probe), outcome,
            self.failure_threshold, self.cooldown))

tei_circuit_breaker = CircuitBreaker("tei-embedding")
llm_circuit_breaker = CircuitBreaker("llm")
vision_circuit_breaker = CircuitBreaker("vision")
rerank_circuit_breaker = CircuitBreaker("rerank")
