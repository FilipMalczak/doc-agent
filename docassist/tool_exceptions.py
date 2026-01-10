
class AgentDoesntKnow(RuntimeError):
    def __init__(self, comment: str):
        self.comment = comment
        super().__init__(f"Agent couldn't provide a meaningful answer ({comment})")

class EmptyResult(RuntimeError):
    def __init__(self, comment: str):
        self.comment = comment
        super().__init__(f"Agent indicated no answer to current task ({comment})")
