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
    
    @property
    def current(self):
        return self.__current
    @current.setter
    def current(self, new_current):
        self.__current = new_current

    def __Add(self, item):
        if (self.__relationToFind != SubsetRelation.Unrelated):
            i = 0
            while i < len(self.__current):

                relation = self.__comparer(item, self.__current[i])
                if (relation == SubsetRelation.Equal or relation == self.__inverseRelation):
                    return
                elif (relation == self.__relationToFind):
                    self.__current.remove(self.__current[i])
                else:
                    i += 1
        
        self.__current.append(item)

    def GetItems(self):
        return self.__current

    def AddRange(self, items):
        for item in items:
            self.__Add(item)