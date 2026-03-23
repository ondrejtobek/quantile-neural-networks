class SignalClass:
    def __init__(self):
        self.Input = {}
        self.Output = []
        self.DependsOn = []

    def CheckOutput(self, Out):
        """
        checks that anomalies in list Out are defined in the class
        """
        if Out[0] == "default":
            Out = self.Output
        if any([not item in self.Output for item in Out]):
            raise Exception(
                "Requested anomaly is not defined in the given class. Out should be a list of characters."
            )
        return Out
