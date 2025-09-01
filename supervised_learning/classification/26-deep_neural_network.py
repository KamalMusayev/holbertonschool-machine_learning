    def save(self, filename):
        """Save Function"""
        filen = filename.split(".")
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load Function"""
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            a = pickle.load(f)
        return a
