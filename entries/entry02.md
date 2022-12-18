# Entry 2: letsBegin.py
## 12/18/22

Over the past few weeks, I have been learning about Python, in order to learn through the courses on visual machine learning that I bought. I have been following the [YouTube Video](https://www.youtube.com/watch?v=cBQ4c1IQJSE), so that over the break, I could start to learn actual machine learning. 

#### **Some of what I have learned so far**

So far, I have learned about Python and just how simple it is. It is a far, far simpler language than anything I have learned before, and it's something that I love. 

For example, Python is known as a *dynamic* and *strong* language. This is because it is dynamic in the fact that you can fluidly transition between data types, so unlike Java for example, I won't need to do any conversion in order to move from one data type to another. In java, if I wanted to find the exact decimal value of `3 / 2`, I would have to write all of the following:
``` java
public class Test {
    public static void main(String[] args) {
        // int's are essentially whole number variables
        int x = 3;
        int y = 2;
        // double's are essentially decimal variables
        double z = (double)(x) / y;
        // need to apply `double` to x, so that the calculations are done as if they are doubles. Without this, the output would result in 1.0 instead of 1.5 
        System.out.println(x + " / " + y + " = " + z);
        // Output: 3 / 2 = 1.5
    }
}
```

But compared that to Python, and this is all you even need:
``` python
x = 3
y = 2
z = 3 / 2
# f'' is just formatting, {} holds your variables. 
print(f'3 / 2 = {z}')
```

This makes writing programs so much easier, because there are less rules to follow. Python is also a strong language because it keeps track of all datatypes, allowing it to be so loose in terms of rules for writing code. 

So far, I haven't learned too much due to other circumstances, but I have managed to learn about *booleans*, *numbers and operations*, and **strings*. 

Strings are very easy to learn, all you have to understand is that this is how they work:
``` python
# No rules about single or double quotes!
first = "Jan"
last = 'Avendano'
print(first + last)
```

For booleans, the only rule is that `True` or `False` must have the first letter capitalized:

``` python
x = True # case sensitive!
y = False

# Boolean comparison operators! ==, !=, <, >, <=, >=
print(f'Equal = {x == y}')
print(f'Not Equal = {x != y}')

print(f'Greater than: {x > y}')
print(f'Greater than or equal to: {x >= y}')
print(f'Less than: {x < y}')
print(f'Less than or equal to: {x <= y}')
# true = 1, and false = 0, so therefore, true > false
```

And for numbers and operations, no conversions are needed!

``` python
# Basic Numerical Operations
x = 3
print(f'x = {x}')

y = x + 3 #add
print(f'add = {y}')

y = x - 1 #subtract
print(f'subtract = {y}')

y = x * 6.86 #multiply
print(f'multply = {y}')

y = x * 0.5 #divide
# Python can cast FOR US!!!
print(f'divide = {y}')

y = x ** 2 #Power
print(f'pow = {y}')

y= x % 2.5 #Remainder
print(f'remainder = {y}')
```

However, there are still different data types, they are just seamlessly convereted to and from. 

``` python
#int
iVal = 34
print(f'iVal = {iVal}')

#float
fVal = 3.14
print(f'fVal = {fVal}')
# Basically, if decimal point needed, use float. If not needed, use int

# Complex - complex([real[, imag]])
cVal = 3 + 6j 
print(f'cVal = {cVal}')
cVal = complex(5, 3)
print(f'real = {cVal.real}, imag = {cVal.imag}')
```

#### **Winter Break Goal**

Over the winter break, I want to use all that I have learned in order to learn as much as I possibly can about machine learning through OpenCV, so that I can begin prototyping and testing my actual project soon. I will do this using the courses I showed in my previous entry, which I have bought to help me learn for my project.

### EDP

I'm definetly still at stage 2 of the **Engineering Design Process**, however as I am learning more about Python and machine learning, I feel I can move on to also brainstorming and planning out my project.

### SKills

A skill I feel I worked on this time is **How to learn**, beacuse of the fact that I have been teaching myself Python through online tutorials. 

I also feel I worked on **Organiziation**, because I have been taking notes all about Python to help walk me through the learning process, as well as for reference later on. 

### Conclusions

I am going to learn as much Python as I possibly can, but once the break starts, I will have to start with machine learning, no matter where I am with Python. I will need to work on my time management skills and really focus on this in order to learn everything I need.

[Previous](entry01.md) | [Next](entry03.md)

[Home](../README.md)