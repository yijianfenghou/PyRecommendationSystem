# # 定义节点类
# class Node(object):
#     def __init__(self, val, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#
#
# # 创建树模型
# node = Node("A", Node("B", Node("D"), Node("E")), Node("C", Node("F"), Node("G")))
#
#
# def BFS(root):
#     # 使用列表作为队列
#     queue = []
#     # 将首个根节点添加队列中
#     queue.append(root)
#     # 将队列不为空时
#     while queue:
#         node = queue.pop(0)
#         left = node.left
#         right = node.right
#         if left:
#             queue.append(left)
#         if right:
#             queue.append(right)
#         print(node.val, end=" ")
#
#
# def DFS(root):
#     # 使用列表作为栈
#     stack = []
#     # 将首个根节点添加
#     stack.append(root)
#     # 当栈不为空时进行遍历
#     while stack:
#         # 从栈的尾部取出一个节点并判断其是否是左右节点
#         # 若有子节点则把对应的子节点压入栈中，且优先判断右节点
#         temp = stack.pop()
#         left = temp.left
#         right = temp.right
#         if right:
#             stack.append(right)
#         if left:
#             stack.append(left)
#         print(temp.val, end=" ")
#
#
# print(BFS(node))
# print(DFS(node))


def isInterleave(s1: str, s2: str, s3: str):
    n, m, t = len(s1), len(s2), len(s3)

    if t != n+m:
        return False

    f = [[False]*(m+1)]*(n+1)
    f[0][0] = True
    print(f)

    for i in range(n+1):
        for j in range(m+1):
            p = i+j-1
            if i > 0:
                print(f[i-1][j])
                print(s1[i-1] == s3[p])
                print("------------")
                f[i][j] = f[i-1][j] and s1[i-1] == s3[p]
            if j > 0:
                f[i][j] = f[i][j-1] and s2[j-1] == s3[p]

    return f[-1][-1]


if __name__ == "__main__":

    print(isInterleave("aabcc", "dbbca", "aadbbcbcac"))
