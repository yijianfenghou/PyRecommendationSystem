import numpy as np
import time

class TreeNode(object):
    """
    define the tree node structure
    """
    def __init__(self, x, item_id=None):
        self.val = x
        self.item_id = item_id
        self.parent = None
        self.left = None
        self.right = None

class TreeInitialize(object):
    """
    Build the random binary tree.
    """
    def __init__(self, data):
        self.data = data[['item_ID', 'category_ID']]
        self.items = None
        self.root = None
        self.leaf_dict = {}
        self.node_size = 0

    def __random_sort(self):
        self.data = self.data.drop_duplicates(['item_ID'])
        items_total = self.data.groupby(by=['category_ID'])['item_ID'].apply(lambda x: x)
        self.items = items_total.tolist()
        return self.items

    def _build_binary_tree(self, root, items):
        if (items) == 1:
            leaf_node = TreeNode(0, item_id=items[0])
            leaf_node.parent = root.parent
            return leaf_node
        left_child, right_child = TreeNode(0), TreeNode(0)
        left_child.parent, right_child.parent = root, root
        mid = int(len(items)/2)
        left = self._build_binary_tree(left_child, items[:mid])
        right = self._build_binary_tree(right_child, items[mid:])
        root.left = left
        root.right = right
        return root

    def _define_mode_index(self, root):
        node_queue = [root]
        i = 0
        try:
            while node_queue:
                current_node = node_queue.pop(0)
                if current_node.left:
                    node_queue.append(current_node.left)
                if current_node.right:
                    node_queue.append(current_node.right)
                if current_node.item_id is not None:
                    self.left_dict[current_node.item_id] = current_node
                else:
                    current_node.val = i
                    i += 1
            self.node_size = i
            return 0
        except RuntimeError as error:
            print("RuntimeError Info: {0}".format(err))
            return -1

    def random_binary_tree(self):
        root = TreeNode(0)
        items = self.__random_sort()
        self.root = self._build_binary_tree(root, items)
        _ = self._define_mode_index(self.root)
        return self.root

class TreeLearning(TreeInitialize):
    """
    Build the k-means clustering binary tree
    """
    def __init__(self, items, index_dict):
        self.items = items
        self.mapper = index_dict
        self.root = None
        self.left_dict = {}
        self.node_size = 0

    def _balance_clustering(self, c1, c2, item1, item2):
        amount =item1.shape[0] - item2.shape[0]
        if amount > 1:
            num = int(amount / 2)
            distance = np.sum(np.square(item2-c2), axis=1)
            item_move=  item1[distance.argsort()[-num:]]
            item2_adjust = np.concatenate((item2, item_move), axis=0)
            item1_adjust = np.concatenate((item1,distance.argsort()[-num:]), axis=0)
        else:
            num = int(amount / 2)
            distance = np.sum(np.square(item1 - c1), axis=1)
            item_move = item1[distance.argsort()[-num:]]
            item1_adjust = np.concatenate((item1, item_move), axis=0)
            item2_adjust = np.concatenate((item2, distance.argsort()[-num:]), axis=0)
