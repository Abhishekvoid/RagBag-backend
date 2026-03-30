import os
import redis
import time

redis_client = redis.from_url(
    os.getenv( "REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True
)

class CircuitBreaker:

    def __init__(self, service_name:str, failure_threshold =3, cooldown=60):
       
        
        self.key = f"cb:{service_name}"
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        

    def is_open(self) -> bool:
        data = redis_client.hgetall(self.key)

        if not data:
            return False  
           
        
        state = data.get("state")
        if state == "open":
            opened_at = float(data.get("opened_at", 0))
            if time.time() - opened_at > self.cooldown:
                redis_client.hset(self.key, "state", "half-open")
                return False
            return True
        return False           
           
    def allow_request(self) -> bool:
        return not self.is_open()
    

    def record_failure(self):
        failures = redis_client.hincrby(self.key, "failures", 1)
        if failures >= self.failure_threshold:
            redis_client.hset(

                self.key,
                mapping={
                    "state": "open",
                    "opened_at": time.time()
                }
            )
    
    def record_success(self):
        redis_client.delete(self.key)

tei_circuit_breaker = CircuitBreaker("tei-embedding")
llm_circuit_breaker = CircuitBreaker("llm")