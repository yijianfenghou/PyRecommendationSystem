# -*- coding: utf-8 -*-
# def quickSort(nums, low, high):
#     if low >= high:
#         return
#     pivot = nums[low]
#     i = low
#     j = high
#     while low < high:
#         while low < high and pivot <= nums[high]:
#             high -= 1
#         nums[low] = nums[high]
#         while low < high and pivot >= nums[low]:
#             low += 1
#         nums[high] = nums[low]
#
#     nums[high] = pivot
#
#     quickSort(nums, i, low-1)
#     quickSort(nums, low+1, j)
#
#
#
# def sortArray(nums):
#     quickSort(nums, 0, len(nums) - 1)
#     return nums


def partition(nums, left, right):
    privot = nums[left]
    while left < right:
        while left < right and nums[right] >= privot:
            right -= 1
        nums[left] = nums[right]
        while left < right and nums[left] <= privot:
            left += 1
        nums[right] = nums[left]

    nums[left] = privot
    return left


def quickSort(nums, left, right):
    if left < right:
        index = partition(nums, left, right)
        quickSort(nums, left, index - 1)
        quickSort(nums, index + 1, right)


def sortArray(nums):
    quickSort(nums, 0, len(nums) - 1)
    return nums


if __name__ == "__main__":



