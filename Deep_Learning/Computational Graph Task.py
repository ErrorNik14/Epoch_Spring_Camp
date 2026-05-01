import numpy as np


# Making the Node class
class Node():
    def __init__(self,value,children=None,op=None,op_args=None):
        self.children = children
        self.op = op
        self.op_args = op_args
        self.value = value

    def backprop(self, gr=1):
        if self.op==None:
            print("Reached the bottom-most nodes!")
            print(self.value, '--->',gr, '\n')
            return gr
        else:
            #print(self.value, self.op,'--->',gr, '\n')
            if self.op_args==None: # We have two nodes
                x,y = self.children[0].value, self.children[1].value
                lg=rg=0 # left and right gradients for further backpropping
                if self.op=='add': # A = x+y
                    lg = 1 # dA/dx = 1
                    rg = 1 # dA/dy = 1
                elif self.op=='sub': # A = x-y
                    lg = 1 # dA/dx = 1
                    rg = -1 # dA/dy = -1
                elif self.op=='mult': # A = x*y
                    lg = y # dA/dx = y
                    rg = x # dA/dy = x
                elif self.op=='div': # A = x/y
                    lg = 1/y # dA/dx = 1/y
                    rg = -x/y**2 # dA/dy = -x/y^2
                elif self.op=='exp': # A = x**y
                    lg = y * x**(y-1) # dA/dx = y * x**(y-1)
                    rg = (x**y)*np.log(x) # dA/dy = (x**y)*lnx

                self.children[0].backprop(gr*lg)
                self.children[1].backprop(gr*rg)

            else: # We have a node and a constant
                x = self.children[0].value
                c,cond = self.op_args
                g=0 # left and right gradients for further backpropping
                if self.op=='add': # A = x+c
                    g = 1
                elif self.op=='sub': # A = x-c or c-x
                    g = 1 if cond else -1
                elif self.op=='mult': # A = x*c
                    g = c
                elif self.op=='div': # A = x/c or c/x
                    g = 1/c if cond else -c/x**2
                elif self.op=='exp': # A = x**c or c**x
                    g = c * x**(c-1) if cond else (c**x)*np.log(c)

                self.children[0].backprop(gr*g)

# Defining the function for arithmetic operations for nodes
def oper(a,b,op):
    nodes = []
    const = None
    v1 = v2 = 0
    val = 0
    if not isinstance(a, Node):
        nodes.append(b)
        const = (a,0)
        v1 = np.array([a])
        v2 = b.value
        #vals = np.array([np.array([a]).reshape(b.value.shape),b.value])
    elif not isinstance(b, Node):
        nodes.append(a)
        const = (b,1)
        v1 = a.value
        v2 = np.array([b])
        #vals = np.array([a.value,np.array([b]).reshape(a.value.shape)])
    else:
        nodes.append(a)
        nodes.append(b)
        v1 = a.value
        v2 = b.value
        #vals = np.array([a.value,b.value])
    #vals = vals.T
    nodes = np.array(nodes)
    if op=='add':
        val = v1+v2
    elif op=='sub':
        val = v1-v2
    elif op=='mult':
        val = v1*v2
    elif op=='div':
        val = v1/v2
    elif op=='exp':
        val = v1**v2
    
    return Node(value=np.array([val]), children=nodes, op=op, op_args=(const if const is not None else None))



a = Node(np.array([1.0, 2.0]))
b = Node(np.array([3.0, 4.0]))

c = oper(oper(a,2,'mult'),b,'add')
d = oper(c,3,'exp')

#e = oper(a,b,'exp')
#f = oper(e,3,'exp')

gradient = d.backprop()
print("GRADIENT: ",gradient)