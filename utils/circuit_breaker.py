
import redis
import time

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

class CircuitBreaker:

    def __init__(self, name, failure_threshold =3, cooldown=60):

        self.key = f"cb:{name}"
        self.failure_threshold = failure_threshold 
        self.cooldown = cooldown    

    def allow_request(self) -> bool:
        data = redis_client.hgetall(self.key)

        if not data:
           
        
            state = data.get("state")

            if state == "open":
                opened_at = float(data.get("opened_at", 0))
                if time.time() - opened_at > self.cooldown:
                    redis_client.hset(self.key, "state", "half-open")
                return True
            return False
        return True
    
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
    
    def record_sucess(self):
        redis_client.delete(self.key)

llm_circuit_breaker = CircuitBreaker("llm")