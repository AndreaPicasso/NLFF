import senticnet_db_5 as data


class Senticnet(object):
    """
    Simple API to use Senticnet 4.
    """
    def __init__(self):
        self.data = data.senticnet

    # public methods

    def concept(self, concept):
        """
        Return all the information about a concept: semantics,
        sentics and polarity.
        """
        result = {}

        result["polarity"] = self.polarity(concept)
        result["sentics"] = self.sentics(concept)
        result["semantics"] = self.semantics(concept)

        return result

    def semantics(self, concept):
        """
        Return the semantics associated with a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        return concept_info[8:]

    def sentics(self, concept):
        """
        Return sentics of a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        sentics = {"pleasantness": float(concept_info[0]),
                   "attention": float(concept_info[1]),
                   "sensitivity": float(concept_info[2]),
                   "aptitude": float(concept_info[3])}

        return sentics

    def polarity(self, concept):
        """
        Return the polarity of a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        return float(concept_info[7])


    def has(self, concept):
        """
        Return if exist that concept in the db
        """
        return concept in self.data
