import numpy as np
class MyCircularQueue(object):

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self.k=k
        self.queue=[None]*k
        self.head=0
        self.tail=None

    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """

        for i in range(len(self.queue)):
            if self.queue[i] is None:

                self.queue[i]=value
                self.tail=i
                print("queue can still be filled")
                #print(self.queue)
                return True
        print(" sorry queue is full")
        #print(self.queue)
        return False

    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        present_element=len([x for x in self.queue if x is not None])
        if present_element==0:
            return False


        self.queue[self.head]=None
        if self.head==self.k-1:

            self.head=0
        else:
            self.head+=1
        return True




    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        return self.queue[self.head]


    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        #print(self.tail)
        #print(self.queue)
        return self.queue[self.tail]

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        present_element=len([x for x in self.queue if x is not None])
        if present_element!=0:
            return False
        else:
            return True

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        present_element=len([x for x in self.queue if x is not None])
        if present_element==self.k:
            return True
        else:
            return False


# Your MyCircularQueue object will be instantiated and called as such:
obj = MyCircularQueue(5)
param_1 = obj.enQueue(23)
param_1 = obj.enQueue(24)
param_1 = obj.enQueue(25)
param_1 = obj.enQueue(26)
param_1 = obj.enQueue(26)
param_1 = obj.enQueue(27)
#print(obj.head)
#print(obj.tail)
'''param_2 = obj.deQueue()
param_2 = obj.deQueue()
param_2 = obj.deQueue()
param_2 = obj.deQueue()
param_2 = obj.deQueue()

param_1 = obj.enQueue(27)
param_1 = obj.enQueue(28)'''

param_3 = obj.Front()
print(param_3)
param_4 = obj.Rear()
print(param_4)
param_5 = obj.isEmpty()
print(param_5)
param_6 = obj.isFull()
print(param_6)
