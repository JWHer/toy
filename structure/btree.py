class Node:
    def __init__(self, data):
        self.data = data
        self.left = self.right = None

    # You can override here to print specific data
    # def __repr__(self):
    #     pass

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, data) -> bool:
        self.root = self._insert(self.root, data)
        return self.root is not None

    def _insert(self, node:Node, data) -> Node:
        if node is None:
            node = Node(data)
        else:
            if node.data >= data:
                node.left = self._insert(node.left, data)
            else:
                node.right = self._insert(node.right, data)
        return node

    def find(self, key) -> bool:
        return self._find(self.root, key) is not None

    def _find(self, node:Node, key) -> Node:
        if node is None or node.data==key:
            return node
        elif node.data > key:
            return self._find(node.left, key)
        else:
            return self._find(node.right, key)

    def delete(self, key) -> bool:
        return self._delete(self.root, key) is not None

    def _delete(self, node:Node, key):
        if node is None:
            return node

        if node.data==key:
            if node.left and node.right:
                # replace node to the left-most of node.right
                parent, child = node, node.right
                while child.left is not None:
                    parent, child = child, child.left
                # child==left-most of node right
                child.left = node.left
                # node.right!=left-most of node right itself
                if parent != node:
                    parent.left = child.right
                    child.right = node.right
                node = child
            elif node.left or node.right:
                node = node.left or node.right
            else:
                node = None
        elif node.data > key:
            node.left = self._delete(node.left, key)
        else:
            node.right = self._delete(node.right, key)
        return node

    def traversal(self, order='in'):
        def _pre_order(root):
            if root is None:
                pass
            else:
                print(root.data)
                _pre_order(root.left)
                _pre_order(root.right)

        def _in_order(root):
            if root is None:
                pass
            else:
                _in_order(root.left)
                print(root.data)
                _in_order(root.right)

        def _post_order(root):
            if root is None:
                pass
            else:
                _post_order(root.left)
                _post_order(root.right)
                print(root.data)
        order = order.lower()
        if order=='pre':
            _pre_order(self.root)
        elif order=='in':
            _in_order(self.root)
        elif order=='post':
            _post_order(self.root)
        else:
            KeyError('Out of order: {}'.format(order))


array = [40, 4, 34, 45, 14, 55, 48, 13, 15, 49, 47]

bst = BinarySearchTree()
for x in array:
    bst.insert(x)

bst.traversal()

# Find
print(bst.find(15)) # True
print(bst.find(17)) # False

# Delete
print(bst.delete(55)) # True
print(bst.delete(14)) # True
print(bst.delete(11)) # False

bst.traversal()
