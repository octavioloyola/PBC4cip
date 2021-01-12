from core.Item import SubsetRelation

class FilteredCollection(object):
    def __init__(self, comparer, relationToFind, resultCollection=None):
        self.__comparer = comparer
        self.__relationToFind = relationToFind
        self.__inverseRelation = SubsetRelation.Unrelated

        if self.__relationToFind == SubsetRelation.Superset:
            self.__inverseRelation = SubsetRelation.Subset
        elif self.__relationToFind == SubsetRelation.Subset:
            self.__inverseRelation = SubsetRelation.Superset
        elif self.__relationToFind == SubsetRelation.Equal:
            self.__inverseRelation = SubsetRelation.Different

        self.__current = None
        if not resultCollection:
            self.__current = list()
        else:
            self.__current = resultCollection

    def SetResultCollection(self, current):
        self.__current = current

    def Add(self, item):
        print(f"item: {item}")
        #print(f"adding")
        
        if (self.__relationToFind != SubsetRelation.Unrelated):
            # for i in range(len(self.__current)):
            # if hasattr(item, 'Items'): print(f"{item.__repr__()}... ", end = ' ')
            i = 0
            print(f"self.__current size: {len(self.__current)}")
            while i < len(self.__current):

                relation = self.__comparer(item, self.__current[i])
                if (relation == SubsetRelation.Equal or relation == self.__inverseRelation):
                    print(f"subsetRelation.Equal")
                    # if hasattr(item, 'Items') and relation == SubsetRelation.Equal: print(f"Not added (equal to {self.__current[i].__repr__()})... ")
                    # if hasattr(item, 'Items') and relation == self.__inverseRelation and relation==SubsetRelation.Subset: print(f"Not added (it is not a superset of{self.__current[i].__repr__()})... ")
                    # if hasattr(item, 'Items') and relation == self.__inverseRelation and relation==SubsetRelation.Superset: print(f"Not added (it is not a subset of{self.__current[i].__repr__()})... ")
                    return
                elif (relation == self.__relationToFind):
                    print(f"relation == self.__relationToFind")
                    # if hasattr(item, 'Items'): print(f"Removing {self.__current[i].__repr__()}... ")
                    # if hasattr(item, 'Items'): print(f"{item.__repr__()}... ", end = ' ')
                    self.__current.remove(self.__current[i])
                else:
                    i += 1
        # if hasattr(item, 'Items'): print(f"Added")
        self.__current.append(item)
        print(f"self.__current {self.__current} size: {len(self.__current)}")

    def GetItems(self):
        return self.__current

    def AddRange(self, items):
        for item in items:
            self.Add(item)
        varitems = self.GetItems()

    def Clear(self):
        self.__current = list()