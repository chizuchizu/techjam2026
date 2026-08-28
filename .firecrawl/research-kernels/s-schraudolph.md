

NOTE
Communicate   d by  Michael Hines
A Fast,Compact Approximatio   n of  th e Exponentia l Function
Nicol N.  Schraudolph
IDSIA,Lugano,Switzerland
Neural network simulations  ofte   n spend a larg   e proportion of thei  r ti me
computing exponential functions. Since th e exponentiation  routines of
typical mat   h libraries ar e rather slo  w,thei   r replacement  wit   h a fas  t ap -
proximation ca n greatly reduce th e overall computation  time. Thi   s article
describes ho w exponentiation  ca n be  approximated  by  manipulating  th e
components of  a standard (IEEE-754) oating-point representation  . Thi   s
models th e exponential function as wel   l as  a lookup tabl   e wit   h linear
interpolatio n,bu t is signicantly faster an d mor   e compact.
1 Motivation
Exponentiation is arguably the quintessential nonlinear function of  neural
computation. Among othe   r use s
,
it is needed to compute mos  t of the activa-
tion functions an d probability distributions used in neural network models.
Consequently
,
muc   h of  the tim e in neural simulations is actually spent on
exponentiation.
Theexpfunctions providedby typicalcomputermat  hlibrariesare highly
accurate but rather slo w. An  approximation is perfectly adequate for mos   t
neuralcomputation purposes an d ca n save muc   h time   . In recognition ofthis
,
man  y neural network software packages approximateexpwith a lookup
table
,
typically with linear interpolation. There is
,
however
,
an even faster
and highly compact way to obtain comparable approximation quality.
2 Th  e Algorithm
Floating-point numbers are typically represented on computers in the for  m
(
¡
1
)
s
(
1
C
m
)
2
x¡x
0
,
where
s
is the sign bit
,m
the mantissa—a binary fraction
in the range
[
0
,
1
)
—and
x
the exponent
,
shifted by a constant bia  s
x
0
. The
widely use d IEEE-754 standard (IEEE
,
1985) species a 52-bit mantissa and
an 11-bit exponent wit  h bias
x
0
D1023
,
laid ou t in 8 bytes of  computer
memory
,
as shown in Figure 1 (top row). The components of  this represen-
tatio   n ma  y be  manipulated by accessing the same memory location as a pai  r
of  4-byte integers (denoted
i
and
j
here). In particular
,
an y integer written
directly to the
x
component (via
i
) wil  l be exponentiated when th e same
memory location is read bac  k in oating-point format. Thi s is the key idea
behind the fas  t exponentiation macro proposed here.
Neural Computation
11,853–862
(1999)
c
°199 9 Massachusetts Institute of Technology

854Nicol N. Schraudolph
sxxxxxxxxxxxmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
12345678
iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiijjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj
Figure1:Bi t representation of the
union
data structure use  d by  the
EXP
macro.
The same 8 byte s ca n be  accessed either as an IEEE-754
double
(to p row  ) with
signs,exponentx,and mantissam,or  as two 4-byte integersiandj(bottom).
Sincethe
x
component resides in the higher-order bits of
i,
an integer
y
to be exponentiated mus  t be left-shifted by 20 bits
,
afte r the bia  s
x
0
has been
added.Thus
i:
D2
20
(
y
C
1023
)
computes2
y
for  integer
y
. No  w consider wha  t
happens for noninteger arguments
:
Afte   r multiplication
,
the fractional par t
of
y
wil  l spil   l ove  r int  o the highest-order bit  sof the mantissa
m
.Thi s spillover
is not onl  y harmless
,
but in fact is highly desirable—under th e IEEE-754
format
,
it amounts to a linear interpolation between neighboring integer
exponents. The technique therefore exponentiates real-valued arguments
as  well as  a lookup table with 2
11
entries and linear interpolation.
Finally
,
to compute
e
y
rather than 2
y
,y
mus   t be divided by ln
(
2
)
rst.  The
complete transformation of
y
necessary to compute a fast approximation to
e
y
in the IEEE-754 format is give   n by
i:
D
a y
C(
b¡c
)
(2.1)
where
a
D2
20
/
ln
(
2
)
,b
D1023
¢
2
20
,
and
c
is an adjustment parameter tha t
affordssom  e controlove rthe propertiesof the approximation (se  e section 4).
Figure 2 shows C cod  e implementing this method. TheLITTLEENDIAN
ag is necessary sinc   e computers differ in how the  y stor   e multibyte quanti-
ties in memory. The simplest way to determine whether it should be set on  a
given machine is to try bot  h alternatives. Theuniondat  a structure should
be declaredstaticto ensure that
j
(which is never used by the macro)
is initialized to zer o
,
as  well as to avoi   d nam  e clashes when this cod  e is
included in multiple source modules
,
for example
,
from a common header
le.
For integer arguments
y,
a signicant additional speedup (see Tabl  e 1)
can be obtained at  littl  e cos  t in accuracy by settingEXPAandEXPCto
integer values
,
so thattheEXPmacro need not perform any oating-point
arithmetic at all. This tric  k can be use d in conjunction wit  h noninteger quan-
tizations as well
:
By  premultiplyingEXPAwit  h the (real-valued) quantum
q,
the n rounding to integer
,
one obtains a macro that approximates
e
yq
for
integer
y,
usin   g only integer arithmetic. However
,
in ou r experience
,
casting
inherently real-valued arguments to integer in order to exploit this feature
is generally
not
a goodidea
,
sinc e type conversion from oating point to
integer tends to be a comparatively expensive operation.

Approximation of the Exponential Function855
#include &lt;math.h&gt;
static union
[
doubled;
struct
[
#ifdef LITTLEÇ ENDIAN
int  j, i;
#else
int  i, j;
#endif
] n;
] Ç eco;
#define EXP Ç A (1048576/MÇ  LN2  )  /* use 1512775 for integer version */
#define EXP Ç C  60801/* see tex  t for choice of  c values */
#define EXP(y) (Ç eco.n.i = EXPÇ A*(y) + (1072693248 - EXP  Ç C), Ç eco.d)
Figure2:C cod e implementing the
union
dat a structure and
EXP
macro for fas  t
approximate exponentiation.
LITTLEENDIAN
mus   t be denedfo r machinestha  t
stor   e integers with th e leas   t signicant byte rst;
EXPC
is set to the desired value
of  thecparameter (see section 4).
Table  1:Seconds Required fo r 10
8
Exponentiations on a Variet  y of Workstations.
ManufacturerIntelSGISunDEC
ProcessorPentiumMIPSUltraSparcAlpha server
Model/SpeedPro/2404600SG1/1702100A/300
LITTLEENDIAN
YesNoNoYes
Op. SystemLinux 2.0.2   9Irix 5.3SunOS 5.5.   1OSF   1 4. 0
Compilergcc  2.7.2.1
/bin/cc
gcc 2.7.2.1D
EC
C 5.2
Optimization-O2-O4-O2-fast
exp
(
libm.a
)8912616628
Lookup tabl   e46622223
EXP
macro28257.64.2
EXP
(integers)6.26.83.7-0.6
3 Benchmark Results
Tabl e 1 list s the benchmark results obtained on  a variety of machines for the
standard mat  h library
’
sexpfunction
,
a lookup tabl   e with linear interpola-
tion
,
and theEXPmacro in its general ( oating-point) and integer forms. The
benchmark program was required to return the su m of 10
8
exponentials of
pseudorandom arguments so as to prevent
“
optimizing awa   y
”
of  any expo-

856Nicol N. Schraudolph
nentiation by the compiler. On each machine
,
the time taken to calculate just
the sum of  the 10
8
pseudorandom arguments was subtracted to obtain net
computing time   s for the exponentiation. To check for variation in th e CPU
time consumed
,
each benchmark was ru n thre e times. The gures shown in
Table 1 are averages ove  r these three runs
;
the observed uctuations wer  e
verysmall.
The results show tha t theEXPmacro is clearly the fastest on  all machines
tested. For oating-point arguments it requires between 18% (DE  C Alpha)
an d 60% (Intel Pentium Pro) of the tim e needed by the lookup table. No  t
surprisingly
,
th e standard mat  h library
’
sexproutines follow fa r behind.
The-fastoptimization switch on the DE  C Alpha activates an approximate
exproutine that is only slightly slower tha n a lookup tabl e
,
but th e othe r
machines do not have such a feature. Performance on the Su n workstation
in particular suffers from anexpfunction that is almost 22 time s slower
thantheEXPmacro.
This discrepancy grows to an  impressive 45-fold speed advantage for
the integer form of the macro. The integer variant is signicantly faster tha n
the general (oating-point) for  m ofEXPon  all tested machines. On th e DE  C
Alpha
,
it appears to be even faster than ligh t
,
taking negative time ! Recall
,
though
,
tha t thes   e gures denote net computing time   s
,
from which the time
take  n by a control—the sam  e program with th e exponentiation removed—
has bee  n subtracted. In this case
,
the integerEXPmacro wa  s on  average 6
nanoseconds faster tha  n the integer to oating-point type conversion tha  t
take  s plac e instead in the control program. Although not as impressive as
violating basic law s of  physics
,
this still testies to a rather astonishing
speed.
In summary
,
these benchmark results indicate that theEXPmacrocould
greatly accelerate computations that mak   e heavyuseof exponentiation.
1
It is
both faster an d mor   ecompact than a lookup tabl  ewit  h linear interpolation
,
a
widelyused acceleration method.Finally
,
it sspeed is eve  ngreaterfo rinteger
arguments
,
as occur
,
for example
,
in the calculation of the Boltzmann-Gibbs
distribution for quantized energy levels.
4 Approximatio n Properties
ComputingEXP(
y
) is very fas  t
,
but ho w well does it approximate
e
y
?Figure3
shows the logistic function implemented using theEXPmacro versus the
standard mat  h library
’
sexpfunction. The left panel illustrates that on a
global scale
,
the two ar e all bu t indistinguishable. Th e greater magnication
in the center panel highlights th e linear interpolation performed byEXP
1
To  giv  e an example,Lazzaro and Wawrzynek’s (1999) neural network-based JPEG
quality transcoder runs twic   e as  fast when using the
EXP
macro (Lazzaro,personal com  -
munication).

Approximation of the Exponential Function857
-4  -2  0   2   4
0.0
0.2
0.4
0.6
0.8
1.0
3.03.5
0.950
0.975
3.30325  3.30326
0.9645405
0.9645400
Figure3:Comparison of  the logistic functiony7!
(
1
C
e
¡y
)
¡1
implemented
usingthe
EXP
macro (soli d line,forcD60,801  ) versus the math library’s
exp
function (dashed line). Different scales highlight the global t (left ),the linear
interpolation (center),and the staircase effect (right).
due to the limited precision of th e 11-bit exponent
x
. Finally
,
the highly
magnied right-hand panel of Figure 3 shows that on  the ver  y small scale
ofD
y
D2
¡20
,
EXP(
y
) exhibits a staircase structure. This happens because
the macro completely ignores the lower par  t
j
of  the mantissa
,
leaving it at
zero—the value to which stati  c variables in C are initialized—for reasons of
ef  ciency.
VersionsofEXPtha t use 8-byte (longlong) integers do  not suffer from
this staircase effect
,
bu t were found to be unacceptably slow on th e typical
workstation platforms. As  it stands
,
EXP(
y
) is thus monotonically nonde-
creasing but (unlike
e
y
) not monotonically increasing. Although this should
be kept in min  d when writing cod e that uses theEXPmacro
,
in practice it
should not present any difculties.
The
c
parameter in equation 2.1 permits some ne-tuning of th e ap -
proximation for certain desirable characteristics. For
c
D0
,
theEXPmacro
interpolates between 2
11
points tha t li e exactly on  the exponential func-
tion
:
EXP
(
n
ln 2
)
D
e
nln 2
D2
n
for all integer
n
. Du  e to the staircase effect
,
however
,
an upper bound on the exponential (
8y
EXP
(
y
)
 ̧
e
y
) requires
c
·
¡
1. Positive values of
c
right-shiftEXP
(
y
)
;
a lower bound on
e
y
is
returned for
c
 ̧90
,
253. (Mathematical derivations for these values ar e
presented in th e appendix.) If tigh t bounds are required on  bot  h sides
,
a
particularly efcien t way to compute them for a give  n argument is to cal  l
the macro
#define EXP Ç L (Ç eco.n.i -= 90254, Ç eco.d),
which returns the lower bound
,
righ t afte  r computing the upper bound by
a call toEXP(withEXPCset to –1). Intermediate values of
c
producethe
best overall approximations
:
th e maximum relative erro  r (to either side of
e
y
) is smallest for
c¼
45
,
799
,
the minimum root-mean-square (RMS) rela-
tive error is reached at
c¼
60
,
801
,
an d the mea  n relative erro  r is lowest at
c¼
68
,
243.(See th e appendix.)

858Nicol N. Schraudolph
Table2:Relative Erro   r of the EXP Macro fo r Various Choices of  thecParameter.
Relative Erro   r:Max.
&lt;
e
y
:Max.
&gt;
e
y
:Root Mea  n Square:Mean:
c ́c¢ln
(
2
)
/
2
20
(
1¡e
¡c
)
(
2e
¡(c
C
1
)
/
ln
(
2
)
¡1
)
(
p
Y
(
c
))(
W
(
c
))
c
D
¡10.000%6.148%4.466%4.069%
c
D
45,7992.982
D
2.9822.0311.811
c
D
60,8013.9391.9661.7701.522
c
D
68,2434.4111.4661.8371.483
c
D
90,2535.7920.0002.6171.959
Table 2 list  s the maximum (below and above
e
y
)
,
RMS
,
and mea  n relative
errorofEXPfor each of the above settings of
c,
with optimal erro  r values
italicized. These values have been measured empirically
;
the y are in perfect
agreement wit  h the analytically derived formulas shown in the column
headings
,
which ste  m from equations A.  7
,
A.8
,
and A.12.
5 Li mitations
TheEXPmacro proposed here provides a ver y fast
,
reasonably accurate
approximationof the exponentialfunction. Nevertheless
,
its speed is bought
at  a price
:
 It requires 4-byte integers and IEEE-754-compliant oating-point dat  a
types. (These ar e available in mos   t computing environments.)
 Its implementation depends on  th e byte order of  the machine.
 Its use of  a global stati   c variable is problematic in multithreaded en-
vironments. (Each thread mus  t have a private cop  y of  theecodata
structure.)
 There is no overow or error handling. The use  r mus  t ensure that the
argument is in the valid range (roughly
,¡
700 to 700).
 It only approximates the exponential function (se  e section 4). Certain
numerical methods ma  y amplify th e approximation error
;
eachalgo-
rithmto useEXPshould therefore be tested against the original version
rst.
In situations where these limitations ar e acceptable
,
theEXPmacro
promises to spee  d up the computation of  exponentials greatly.

Approximation of the Exponential Function859
Appendix:Mathematical Analysis
Ignoring the staircase effect shown in Figure 3 (right)
,
theEXPmacrocan  be
described as
EXP
(
y
C
c
)
D2
k
(
1
C
y
/
ln
(
2
)
¡k
)
,
where
k ́by
/
ln
(
2
)
c,
(A.1)
wherec
 ́c¢
ln
(
2
)
/
2
20
,
and
buc
denotes the largest integer·
u
. In what
follows
,
various values of
c
ar e derived for which equation A.  1 ha s certain
desirable properties.
A.  1 Upper and Lower Bound.The exponential inequality states that
:
2
a
·1
C
a
8
a
2[
0
,
1
]
2
y
/
ln
(
2
)
¡k
·1
C
y
/
ln
(
2
)
¡k
e
y
·EXP
(
y
C
c
)
.(A.2)
Forc·0 this implies
e
y
·EXP
(
y
)
. Th e corresponding bound on
c
mustbe
decremented by 1 on account of the staircase effect
;
theEXPmacrohence
returns an  upper bound to the exponential function for
c
·
¡
1.
To determine th e smallest valu   e of
c
for which EXP
(
y
)
delivers a lower
boundto
e
y
,
match the tw o functions
’
rs t derivatives
:
@
@y
e
y
C
c
D
@
@y
EXP
(
y
C
c
)
e
y
C
c
D2
k
/
ln
(
2
)
y
C
cD
k
ln
(
2
)
¡
ln
(
ln
(
2
))
y
/
ln
(
2
)
¡k
D
¡[
ln
(
ln
(
2
))C
c
]
/
ln
(
2
)
.(A.3)
Then compare function values at th e points characterized by equation A.  3
:
e
y
C
c
 ̧EXP
(
y
C
c
)
2
k
/
ln
(
2
)
 ̧2
k
(
1
C
y
/
ln
(
2
)
¡k
)
1 ̧ln
(
2
)
¡[
ln
(
ln
(
2
))C
c
]
c
 ̧2
20
[
1
¡[
ln
(
ln
(
2
))C
1
]
/
ln
(
2
)
]¼
90
,
252  .34.(A.4)
Rounding up to preserve the bound yields the best integer value of
c
D
90
,
253.
A.  2 Lowest Maximum Relative Error..For intermediate values of
c,
EXP dip s bot  h above and below the exponential function. The relative error
is greatest at  the extrema of
r
c
(
y
)
 ́
1
¡
EXP
(
y
C
c
)
/
e
y
C
c
.(A.5)

860Nicol N. Schraudolph
Setting it s derivative to zer  o
,
@
@y
r
c
(
y
)
D
2
k
(
1
C
y
/
ln
(
2
)
¡k
)
¡
2
k
/
ln
(
2
)
e
y
C
c
D0
y
D
(
k¡
1
)
ln
(
2
)C
1
,
(A.6)
yields the loca  l minima of
r
c
(
y
)
. The loca   l maxima ca n be found at the points
where EXP is not differentiable
,
that is
,
at
y
D
k
ln
(
2
)
. Th e maximum relative
error is lowest when the magnitude of
r
c
(
y
)
is equal at both sets of extrema
:
|
r
c
[k
ln
(
2
)
]
|
D
|
r
c
[
(
k¡
1
)
ln
(
2
)C
1
]
|
1
¡e
¡
c
D2
e
¡
(
c
C
1
)
/
ln
(
2
)
¡
1
cDln
(
ln
(
2
)C
2
/
e
)
¡
ln
(
2
)
¡
ln
(
ln
(
2
))
c
Dc
¢
2
20
/
ln
(
2
)
¼
45
,
799  .12(A.7)
The staircase effect ca n be adjusted for by subtracting 0.5 from this valu   e
;
the best integer choice is
c
D45
,
799.
A.  3 Lowest RM  S Relative Error.To compute the valu   e of
c
thatmini-
mizes the RM S relative erro  r
,
consider th e integrated squared relative erro  r
Y
:
Y
(
c
)
 ́
1
2
n
ln
(
2
)
Z
nln
(
2
)
¡nln
(
2
)
r
c
(
y
)
2
dy
D
1
2
n
ln
(
2
)
n¡1
X
i
D
¡n
Z
(
i
C
1
)
ln
(
2
)
iln
(
2
)
³
1
¡
2
i
[
1
C
y
/
ln
(
2
)
¡i]
e
y
C
c
 ́
2
dy
D
¢¢¢
D1
C
3
C
4
(
1
¡
4
e
c
)
ln
(
2
)
16
e
2
c
ln
(
2
)
3
.(A.8)
Setting the derivative ofYto zer  o gives
:
@
@
c
Y
(
c
)
D
4
(
2
e
c
¡
1
)
ln
(
2
)
¡
3
8
e
2
c
ln
(
2
)
3
D0
2
e
c
¡
1D
3
4 ln
(
2
)
c
D2
20
ln
³
3
8 ln
(
2
)
C
1
2
 ́
/
ln
(
2
)
¼
60
,
801 .48 .(A.9)
Again 0.5 mus  t be  subtracted to compensate for the staircase effect
;
the best
integer valu   e is
c
D60
,
801.

Approximation of the Exponential Function861
A.  4 Lowest Mean Relative Error..The points at  which EXP intersects
the exponential function are given by
e
y
C
c
DEXP
(
y
C
c
)
e
y
e
c
D2
k
(
1
C
y
/
ln
(
2
)
¡k
)
¡e
c
ln
(
2
)
/
2D
[k
ln
(
2
)
¡y¡
ln
(
2
)
]e
kln
(
2
)
¡y¡ln
(
2
)
W
(
¡e
c
ln
(
2
)
/
2
)
D
k
ln
(
2
)
¡y¡
ln
(
2
)
y
/
ln
(
2
)
¡k
D
¡W
(
¡e
c
ln
(
2
)
/
2
)
/
ln
(
2
)
¡
1
,
(A.10)
where
W
denotes Lambert
’
s function (Fritsch
,
Shafer
,
&amp; Crowley
,
1973
;
Cor-
less
,
Gonnet
,
Hare
,
&amp; Jeffrey
,
1993
;
Corless
,
Gonnet
,
Hare
,
Jeffrey
,
&amp; Knuth
,
1996)
,
2
which satises
W
(
u
)
e
W
(
u
)
D
u
. Each linear segment of  EXP crosses
the exponential at  tw o points
,
r
C
andr
¡
,
given by  the two real-valued
branches
,W
0
and
W
¡1
,
of  Lambert
’
s function
:
r
C|
¡
 ́¡W
0
|
¡1
(
z
)
/
ln
(
2
)
¡
1
,
where
z ́¡e
c
ln
(
2
)
/
2.(A.1  1)
The mea  n relative errorWas  a function ofccan be computed by splitting
the integral ove r the relative erro  r
|
r
c
(
y
)|
at  the crossover pointsr
C|
¡
:
W
(
c
)
 ́
1
2
n
ln
(
2
)
Z
nln
(
2
)
¡nln
(
2
)
|
r
c
(
y
)|
dy
D
1
2
n
ln
(
2
)
n¡1
X
i
D
¡n
2
4
Z
(
i
C
r
C
)
ln
(
2
)
iln
(
2
)
r
c
(
y
)
dy¡
Z
(
i
C
r
¡
)
ln
(
2
)
(
i
C
r
C
)
ln
(
2
)
r
c
(
y
)
dy
C
Z
(
i
C
1
)
ln
(
2
)
(
i
C
r
¡
)
ln
(
2
)
r
c
(
y
)
dy
3
5
D
¢¢¢
D1
C
2
ln
(
2
)
μ
W
¡1
(
z
)
2
C
1
W
¡1
(
z
)
¡
W
0
(
z
)
2
C
1
W
0
(
z
)
¶
¡
e
¡
c
2 ln
(
2
)
2
.  (A.12)
Setting th e derivative ofWto zero give  s
@
@
c
W
(
c
)
D4 ln
(
2
)
[W
¡1
(
z
)
¡W
0
(
z
)
]
C
e
¡
c
W
¡1
(
z
)
W
0
(
z
)
D0
e
¡
c
D4 ln
(
2
)
μ
1
W
¡1
(
z
)
¡
1
W
0
(
z
)
¶
1
/
8D
e
W
0
(
z
)
¡e
W
¡
1
(
z
)
.(A.13)
Nowset
º
C|
¡
 ́e
W
0
|
¡
1
(
z
)
. By denition
,W
(
z
)
e
W
(
z
)
D
z
for all branches of
W,
so
z
D
º
C
ln
(
º
C
)
D
º
¡
ln
(
º
¡
)
. In conjunction with equation A.13
,
this yields
º
D
(
º
C
1
/
8
)
ln
(
º
C
1
/
8
)
/
ln
(
º
)
,
(A.14)
2
I have written Octave/Matlab cod e that evaluates any branchof Lambert’sWfunction
for complex arguments. It is available on the Internet at:ftp://ftp.idsia.ch/pub/nic/W.m.

862Nicol N. Schraudolph
which can be solved numerically by iterating ove r equation A.14 from a
suitable starting point 0
&lt;º
0
&lt;
7
/
8. The result is
º¼
0.3071517227
z
D
º
ln
(
º
)
¼¡
0.362566022
cDln
(
¡
2
z
)
¡
ln
(
ln
(
2
))
¼
0.045111411
c
Dc
¢
2
20
/
ln
(
2
)
¼
68
,
243.43.(A.15)
Wit  h th e usual subtraction of  0.5 on account of  the staircase effect
,
the best
integer valu   e is
c
D68
,
243.
Acknowledgments
I than   k Avrama Blackwell
,
Frank Dellaert
,
FelixGers
,
an d the anonymous
reviewers for their helpful suggestions
,
and th e developers of  the Maple
computer algebra system for creating suc h a usefultool  . Lee Campbell of the
Computational Neurobiology Lab at th e Sal k Institute has graciously pro-
vide  d access to and information about a variety of workstations for bench-
marking purposes. This wor   k wa  s supported by the Swiss National Science
Foundation under grant numbers 2100–045700.95/1 and 2000–052678.97/1.
References
Corless,R. M.,Gonnet,G. H.,Hare,D. E. G.,&amp; Jeffrey,D. J. (1993). Lambert’s W
function in Maple.Maple Technical Newsletter,9,12–  22.
Corless,R. M.,Gonnet,G. H.,Hare,D. E. G.,Jeffrey,D. J.,&amp; Knuth,D. E. (1996).
On the Lambert W function.Advances in Computational Mathematics,5(4),
329– 359.
Fritsch,F. N.,Shafer,R. E.,&amp; Crowley,W. P. (1973). Algorithm 443:Solution of the
transcendental equationw e
w
Dx.Communications of the ACM,16,123– 124.
IEEE. (1985).Standard fo r binary oating-point arithmetic. ANSI/IEE E Std  . 754–
1985 . New York:American National Standards Institute/Institute of Electri-
cal and Electronic Engineers.
Lazzaro,J.,&amp; Wawrzynek,J. (1999). JPE  G quality transcoding usin g neural net-
works trained with a perceptual erro   r measure.Neural Computation,11(1).
Received March 13,1998;accepted Jul y 2,1998.