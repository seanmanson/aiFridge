aiFridge
========

Decision-making algorithm for a smart fridge, created for
UQ's 2015 COMP3702 AI course.

Given a specified input file containing a matrix of potential
items and their purchase frequencies per volume, this algorithm
can run over a period of weeks determining which items should be
bought to satisfy the owner for these weeks. This is run in a simulator
which checks how successful the AI is at predicting the user over
several trial runs.

The AI works by modelling the problem as a Markov decision process,
where the state is given by the items in the fridge, and the transitions
by what the user chooses to eat after a purchase is made for the week.

For small problems, policy iteration is used to determine a solution
for which items are always best to purchase. For larger problems (where
there are many items which can fit in the fridge), the AI is able to
dynamically adapt to using Monte Carlo tree search to deeply examine
purchase history and choose a good policy within a limited time period.


The AI code itself is split into two parts: a problem part, containing the
simulator which tests the AI, and the AI itself, which is used by the simulator.
This split demonstrates that the AI isn't 'cheating' and simply telling the simulator
what to buy.

Installation
------------

Use Ant on build.xml to build a .jar file, which can be used to run the algorithm.
Alternatively, import into eclipse using the build.xml file as a build script.

Usage
-----
```
jar a2-3702.jar inputFileName outputFileName
```

Input/Output formats
--------------------

The input file format is given as follows, with newlines between each piece of information:
```
numberofweeks
costoffailure
discountfactor
fridgetype
probabilitymatrixes
```
Where:
 - numberofweeks = Number of weeks to run simulation over
 - costoffailure = Penalty added for missing an item the user wants to eat that week. The less points, the better.
 - discountfactor = Penalty decay over time. Total penalty is given by the most recent weekly penalty + discountfactor * prevous total. At 0 discount, only the last week matters for penalty. At 1, every week matters equally. At 0.5, this week makes up 50%, the previous makes up 25%, and so on.
 - fridgetype = One of 'tiny', 'small', 'medium', 'large', 'super'.

A probability matrix is a matrix stating the likelihood of the user buying certain
quantities of each item. The input file has X of these matrices listed in order, where X 
is the number of purchasable types of items (given by fridge type).

For a given item, its consumption probability matrix is given by:
```
chanceofeating0at0 [chanceofeating1at0] [chanceofeating2at0] [...]
[chanceofeating0at1] [...]
[chanceofeating0at2]
[...]
```
Each row states how much of this item is currently in the fridge; row 0 = none of this item,
row 2 = two of this item, etc. The numbers across then state the probability of buying 0,
1, 2, or more of this item. These horizontal probabilities must add to 1.
The width of the matrix is given by the max number of a single type which can fit in the fridge.
The height is the capacity of the fridge to fit that many items.


The output file format is simply
```
numberofweeks
shoppingweek1
[shoppingweek2]
[...]
totalpenalty
```
The shopping values are simply space-deliniated lists of how many of each item were bought that
week.