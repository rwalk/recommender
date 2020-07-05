import warnings
import heapq

class BaseRecommender:

    def __init__(self, items):
        self._items = items

    def _top(self, scoreables, scorer, n=10):
        '''
        Compute the top n items by score efficiently
        :param scoreables: a list of items that can be scored with scorer
        :param scorer: a function that takes an item from the list of scorables and returns a score
        :return: sorted list of tuples of item and score
        '''
        return heapq.nlargest(n, scoreables, key=lambda x: scorer(x))

    def _item_index_lookup(self, **kwargs):
        '''
        A very simple metdata handling interface, allowing lookup of index values based on metadata properties
        :param kwargs:
        :return:
        '''
        indexes = []
        for item in self._items:
            match = False
            for k, v in kwargs.items():

                # check each condition
                field_value = item.get(k)
                value = v.lower()

                if field_value:
                    if type(field_value) is str:
                        if value in field_value.lower():
                            match = True
                        else:
                            match = False
                            break
                    elif type(field_value) is list and all([type(f) is str for f in field_value]):
                        if any([value in f.lower() for f in field_value]):
                            match = True
                        else:
                            match = False
                            break
                    else:
                        raise ValueError(f"Field {k} is not queryable!")
                else:
                    match = False
                    break

            if match:
                print(f"Matched item: {item}")
                indexes.append(item["index"])
                if len(indexes) > 25:
                    warnings.warn("More than 25 items matched this query. Only taking first 10.")
                    return indexes
        return indexes