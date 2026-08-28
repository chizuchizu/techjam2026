##### Report GitHub Issue

×

Title: 

Content selection saved. Describe the issue below:

Description:

Submit without GitHub Submit in GitHub

![](https://arxiv.org/static/base/1.0.1/images/icons/smileybones-small.svg) arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about)
 ×

 [![arXiv logo](https://arxiv.org/static/base/1.0.1/images/arxiv-logo-primary-light.svg) Back to arXiv](https://arxiv.org/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html)
 [Report Issue](https://arxiv.org/html/2504.05068v1# "Report an Issue")
 [Back to Abstract](https://arxiv.org/abs/2504.05068v1 "Back to abstract page")
 [Download PDF](https://arxiv.org/pdf/2504.05068v1 "Download PDF")
[](javascript:toggleNavTOC(); "Toggle navigation")
[](javascript:toggleReadingMode(); "Disable reading mode, show header and footer")

1.  [Abstract.](https://arxiv.org/html/2504.05068v1#abstract1 "In Global approximations to the error functionof real argument for vectorized computation")
    
2.  [1 Introduction](https://arxiv.org/html/2504.05068v1#S1 "In Global approximations to the error functionof real argument for vectorized computation")
    
3.  [2 Approximations](https://arxiv.org/html/2504.05068v1#S2 "In Global approximations to the error functionof real argument for vectorized computation")
    
4.  [3 Computations](https://arxiv.org/html/2504.05068v1#S3 "In Global approximations to the error functionof real argument for vectorized computation")
    
5.  [4 Conclusions](https://arxiv.org/html/2504.05068v1#S4 "In Global approximations to the error functionof real argument for vectorized computation")
    
6.  [References](https://arxiv.org/html/2504.05068v1#bib "In Global approximations to the error functionof real argument for vectorized computation")
    

[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2504.05068v1 \[physics.chem-ph\] 07 Apr 2025

Global approximations to the error function  
of real argument for vectorized computation
=========================================================================================

Dimitri N. Laikov Address: Chemistry Department, Moscow State University, 119991 Moscow, Russia Email address: [laikov@rad.chem.msu.ru; dimitri\_laikov@mail.ru](mailto:laikov@rad.chem.msu.ru;%20dimitri_laikov@mail.ru)

Date: August 24, 2026

###### Abstract.

The error function of real argument can be uniformly approximated to a given accuracy by a single closed-form expression for the whole variable range either in terms of addition, multiplication, division, and square root operations only, or also using the exponential function. The coefficients have been tabulated for up to 128-bit precision. Tests of a computer code implementation using the standard single- and double-precision floating-point arithmetic show good performance and vectorizability.

###### Key words and phrases: 

error function, numerical approximation

###### 2020 Mathematics Subject Classification

Primary 33B20, 65D20; Secondary 33F05, 33-04

1\. Introduction
----------------

The error function \[[6](https://arxiv.org/html/2504.05068v1#bib.bib6)\
\] of real argument

|     |     |     |     |
| --- | --- | --- | --- |
| (1.1) |     | erf⁡(x)\=2π​∫0xexp⁡(−x2)​𝑑x\\erf(x)=\\frac{2}{\\sqrt{\\pi}}\\int\_{0}^{x}\\exp\\left(-x^{2}\\right)\\,\\mathrm{d}x |     |

shows up in many mathematical models of physical and other phenomena (far too many even to be listed here), and its numerical evaluation can be a bottleneck of a computational simulation. Standard mathematical libraries of C and FORTRAN implement it since at least 2008, and use diverse approximations for some defined ranges of xx, favoring accuracy over speed, but it can be helpful to have a faster though slightly less accurate implementaion. The vector instructions of modern processors promise a speedup of up to 16 times, but then a branch-free code is needed to harness them. In some physical models \[[1](https://arxiv.org/html/2504.05068v1#bib.bib1)\
, [5](https://arxiv.org/html/2504.05068v1#bib.bib5)\
\] the error function divided by its argument, a well-behaved even function

|     |     |     |     |
| --- | --- | --- | --- |
| (1.2) |     | F0​(x)\=erf⁡(x)xF\_{0}(x)=\\frac{\\erf(x)}{x} |     |

should be carefully evaluated.

We have found global closed-form approximations to both functions ([1.1](https://arxiv.org/html/2504.05068v1#S1.E1 "In 1. Introduction ‣ Global approximations to the error functionof real argument for vectorized computation")
) and ([1.2](https://arxiv.org/html/2504.05068v1#S1.E2 "In 1. Introduction ‣ Global approximations to the error functionof real argument for vectorized computation")
) in terms of addition, multiplication, division, and square root — with or without also using the exponential function — where the accuracy can be systematically improved by taking more polynomial terms with optimized coefficients, reaching 128 bits of precision and stopping there.

We confess having found our approximation formulas by general mathematical arguments using the natural intelligence of our own mind, but then we had to make sure this had not been done before. We see in the literature that the approximations to the error function have been developed since the early days of computation \[[9](https://arxiv.org/html/2504.05068v1#bib.bib9)\
, [7](https://arxiv.org/html/2504.05068v1#bib.bib7)\
, [8](https://arxiv.org/html/2504.05068v1#bib.bib8)\
, [10](https://arxiv.org/html/2504.05068v1#bib.bib10)\
, [2](https://arxiv.org/html/2504.05068v1#bib.bib2)\
, [11](https://arxiv.org/html/2504.05068v1#bib.bib11)\
\], but ours still seems to be new. (We cannot review all such works here as it may grow into a study in the psychology of mathematics.)

2\. Approximations
------------------

We begin with a transformation of the error function ([1.1](https://arxiv.org/html/2504.05068v1#S1.E1 "In 1. Introduction ‣ Global approximations to the error functionof real argument for vectorized computation")
)

|     |     |     |     |
| --- | --- | --- | --- |
| (2.1) |     | erf⁡(x)\=xx2+ϕ⁡(x2)\\erf(x)=\\frac{x}{\\sqrt{x^{2}+\\phi\\left(x^{2}\\right)}} |     |

in terms of a new function ϕ⁡(s)\\phi(s), the xx in the numerator in ([2.1](https://arxiv.org/html/2504.05068v1#S2.E1 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) makes it ideal also for the function ([1.2](https://arxiv.org/html/2504.05068v1#S1.E2 "In 1. Introduction ‣ Global approximations to the error functionof real argument for vectorized computation")
). Looking at its explicit form

|     |     |     |     |
| --- | --- | --- | --- |
| (2.2) |     | ϕ⁡(s)\=s\[erf⁡(s)\]2−s,\\phi(s)=\\frac{s}{\\bigl\[\\erf\\bigl(\\sqrt{s}\\bigr)\\bigr\]^{2}}-s, |     |

one may be misled into thinking it is not good for approximations, but it is. We need ϕ⁡(s)\\phi(s) only for s≥0s\\geq 0 where it is monotonically decreasing, starting from

|     |     |     |     |
| --- | --- | --- | --- |
| (2.3) |     | ϕ⁡(0)\=π4,\\phi(0)=\\frac{\\pi}{4}, |     |

with the negative first derivative

|     |     |     |     |
| --- | --- | --- | --- |
| (2.4) |     | ϕ′​(0)\=π6−1,\\phi^{\\prime}(0)=\\frac{\\pi}{6}-1, |     |

and all the way to the asymptotic limit

|     |     |     |     |
| --- | --- | --- | --- |
| (2.5) |     | lims→∞ϕ⁡(s)\=2π​s​exp⁡(−s).\\lim\_{s\\to\\infty}\\phi(s)=\\frac{2}{\\sqrt{\\pi}}\\sqrt{s}\\exp(-s). |     |

It is natural to further transform

|     |     |     |     |
| --- | --- | --- | --- |
| (2.6) |     | ϕ⁡(s)\=ψ⁡(s)​exp⁡(−s),\\phi(s)=\\sqrt{\\psi(s)}\\,\\exp(-s), |     |

so that for the new function ψ⁡(s)\\psi(s) the rational approximation

|     |     |     |     |
| --- | --- | --- | --- |
| (2.7) |     | ψN​(s)\=∑m\=0N+1Am​N​sm1+∑n\=1NBn​N​sn≈ψ⁡(s)\\psi\_{N}(s)=\\frac{\\sum\_{m=0}^{N+1}A\_{mN}s^{m}}{1+\\sum\_{n=1}^{N}B\_{nN}s^{n}}\\approx\\psi(s) |     |

can be made. The conditions ([2.3](https://arxiv.org/html/2504.05068v1#S2.E3 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
), ([2.4](https://arxiv.org/html/2504.05068v1#S2.E4 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
), and ([2.5](https://arxiv.org/html/2504.05068v1#S2.E5 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) now become

|     |     |     |     |
| --- | --- | --- | --- |
| (2.8) |     | ψ⁡(0)\=π216,\\psi(0)=\\frac{\\pi^{2}}{16}, |     |

|     |     |     |     |
| --- | --- | --- | --- |
| (2.9) |     | ψ′​(0)\=5​π224−π2\=(5​π−12)​π24,\\psi^{\\prime}(0)=\\frac{5\\pi^{2}}{24}-\\frac{\\pi}{2}=\\frac{(5\\pi-12)\\pi}{24}, |     |

|     |     |     |     |
| --- | --- | --- | --- |
| (2.10) |     | lims→∞ψ⁡(s)\=4π​s,\\lim\_{s\\to\\infty}\\psi(s)=\\frac{4}{\\pi}s, |     |

and the rational function ([2.7](https://arxiv.org/html/2504.05068v1#S2.E7 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) can be easily constrained to satisfy them.

Knowing that the exponential function of negative arguments can be approximated, to a given uniform absolute accuracy, by the expression

|     |     |     |     |
| --- | --- | --- | --- |
| (2.11) |     | exp⁡(−s)≈(1+∑n\=1N(2−K​s)n​bn/n!)−2K\\exp(-s)\\approx\\textstyle\\left(1+\\sum\_{n=1}^{N}(2^{-K}s)^{n}b\_{n}/n!\\right)^{-2^{K}} |     |

with either exact bn\=1b\_{n}=1 or optimized bn≈1b\_{n}\\approx 1, and with the right KK and NN, we have sought the approximations to the error function, to a given relative accuracy, in terms of arithmetic and square root operations only. We have ended up finding the approximations

|     |     |     |     |
| --- | --- | --- | --- |
| (2.12) |     | ϕM​N(K)​(s)\=(∑m\=0MAm​M​N(K)​sm1+∑n\=1NBn​M​N(K)​sn)2K≈ϕ⁡(s)\\phi\_{MN}^{(K)}(s)=\\left(\\frac{\\sum\_{m=0}^{M}A\_{mMN}^{(K)}s^{m}}{1+\\sum\_{n=1}^{N}B\_{nMN}^{(K)}s^{n}}\\right)^{2^{K}}\\approx\\phi(s) |     |

to work strikingly well for the right KK, MM, and NN, even without satisfying ([2.5](https://arxiv.org/html/2504.05068v1#S2.E5 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
).

The coefficients in ([2.7](https://arxiv.org/html/2504.05068v1#S2.E7 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) and ([2.12](https://arxiv.org/html/2504.05068v1#S2.E12 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) can be optimized to minimize the maximum

|     |     |     |     |
| --- | --- | --- | --- |
| (2.13) |     | E\=max0<x<∞⁡\|ε⁡(x)\|E=\\max\\limits\_{0<x<\\infty}\\bigl\|\\varepsilon(x)\\bigr\| |     |

relative error

|     |     |     |     |
| --- | --- | --- | --- |
| (2.14) |     | ε⁡(x)\=f⁡(x)erf⁡(x)−1\\varepsilon(x)=\\frac{f(x)}{\\erf(x)}-1 |     |

of the approximation f⁡(x)f(x) based on ([2.1](https://arxiv.org/html/2504.05068v1#S2.E1 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) with ϕ⁡(s)\\phi(s) either from ([2.6](https://arxiv.org/html/2504.05068v1#S2.E6 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) and ([2.7](https://arxiv.org/html/2504.05068v1#S2.E7 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) or from ([2.12](https://arxiv.org/html/2504.05068v1#S2.E12 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
). In practice, this can be done by solving the system of equations

|     |     |     |     |
| --- | --- | --- | --- |
| (2.15) |     | {ε⁡(xi)\=−ε⁡(xi+1),xi<xi+1,i\=1,…,L,ε′​(xi)\=0,i\=1,…,L+1,\\left\\{\\begin{array}\[\]{ccrll}\\varepsilon(x\_{i})&=&-\\varepsilon(x\_{i+1}),&x\_{i}<x\_{i+1},&i=1,\\dots,L,\\\\ \\varepsilon^{\\prime}(x\_{i})&=&0,&&i=1,\\dots,L+1,\\end{array}\\right. |     |

for LL variables: L\=2​N−1L=2N-1 for ([2.7](https://arxiv.org/html/2504.05068v1#S2.E7 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) with ([2.8](https://arxiv.org/html/2504.05068v1#S2.E8 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
), ([2.9](https://arxiv.org/html/2504.05068v1#S2.E9 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
), and ([2.10](https://arxiv.org/html/2504.05068v1#S2.E10 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
); or L\=M+N−1L=M+N-1 for ([2.12](https://arxiv.org/html/2504.05068v1#S2.E12 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) with ([2.3](https://arxiv.org/html/2504.05068v1#S2.E3 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) and ([2.4](https://arxiv.org/html/2504.05068v1#S2.E4 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
). The starting values of the parameters can be taken first from the minimization of the least-squares (p\=1p=1)

|     |     |     |     |
| --- | --- | --- | --- |
| (2.16) |     | E(p)\=∫0∞(ε⁡(x))2​p​𝑑xE^{(p)}=\\int\\limits\_{0}^{\\infty}\\bigl(\\varepsilon(x)\\bigr)^{2p}\\,\\mathrm{d}x |     |

or more general (p\=2,3,…p=2,3,\\dots) functional.

3\. Computations
----------------

We have written a computer code to determine the approximation coefficients using multiple-precision floating-point arithmetic, typically 256 bits. For the exponential-based approximation ([2.7](https://arxiv.org/html/2504.05068v1#S2.E7 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) we have found well-behaved solutions, with all coefficients positive, for up to N\=27N=27, but failed for N\=17,21,25N=17,21,25 where some Bn​N<0B\_{nN}<0. Table [1](https://arxiv.org/html/2504.05068v1#S3.T1 "Table 1 ‣ 3. Computations ‣ Global approximations to the error functionof real argument for vectorized computation")
 shows the accuracy of these approximations given as −log2⁡E\-\\log\_{2}E, the number of significant bits, when the computation is done with a much higher bit precision, and we see an exponential convergence with NN. For the exponential-free approximation ([2.12](https://arxiv.org/html/2504.05068v1#S2.E12 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")
) we do not claim to have worked through all combinations of (M,N,K)(M,N,K), nevertheless we have found 55 well-behaved solutions some of which are shown in Table [1](https://arxiv.org/html/2504.05068v1#S3.T1 "Table 1 ‣ 3. Computations ‣ Global approximations to the error functionof real argument for vectorized computation")
 alongside the exponential-based solutions of comparable accuracy.

Table 1. Accuracy of approximations.

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| exponential-based |     | exponential-free |     |     |     |
| Eq. ([2.7](https://arxiv.org/html/2504.05068v1#S2.E7 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")<br>) |     | Eq. ([2.12](https://arxiv.org/html/2504.05068v1#S2.E12 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation")<br>) |     |     |     |
| NN  | −log2⁡E\-\\log\_{2}E | MM  | NN  | KK  | −log2⁡E\-\\log\_{2}E |
| 1   | 11.0 | 0   | 3   | 1   | 11.5 |
| 2   | 17.6 | 0   | 4   | 2   | 16.7 |
| 3   | 24.2 | 0   | 5   | 2   | 22.7 |
| 4   | 29.9 | 3   | 5   | 6   | 29.6 |
| 5   | 34.0 | 2   | 8   | 3   | 33.8 |
| 6   | 40.5 | 3   | 10  | 3   | 40.2 |
| 7   | 42.4 | 5   | 8   | 5   | 41.9 |
| 8   | 48.3 | 4   | 12  | 3   | 47.4 |
|     |     | 6   | 10  | 5   | 52.2 |
| 9   | 53.9 | 7   | 10  | 6   | 53.7 |
| 10  | 60.1 | 8   | 12  | 6   | 57.9 |
| 11  | 62.0 | 8   | 13  | 5   | 58.6 |
| 12  | 64.7 | 9   | 14  | 6   | 64.0 |
| 13  | 70.7 | 10  | 15  | 6   | 68.3 |
| 14  | 76.0 | 11  | 17  | 5   | 75.0 |
| 15  | 81.0 | 13  | 18  | 6   | 80.4 |
| 16  | 86.3 | 14  | 20  | 6   | 88.6 |
| 18  | 91.0 | 15  | 20  | 6   | 90.6 |
| 19  | 96.5 | 17  | 20  | 8   | 93.5 |
| 20  | 101.2 | 17  | 22  | 8   | 99.8 |
|     |     | 17  | 23  | 7   | 102.3 |
| 22  | 108.1 | 19  | 25  | 7   | 106.3 |
| 23  | 113.6 | 22  | 28  | 8   | 117.2 |
| 24  | 119.8 | 23  | 28  | 8   | 121.5 |
| 26  | 125.4 | 25  | 30  | 8   | 123.2 |
| 27  | 130.1 | 25  | 31  | 8   | 130.2 |

Remarkably, both approximations need almost the same number of polynomial terms to reach a given accuracy. Thus the latter can be faster as KK multiplications are faster than the exponential function, but the former is still useful if the values of both erf⁡(x)\\erf(x) and exp⁡(−x2)\\exp\\left(-x^{2}\\right) are needed.

To study the effects of finite-precision arithmetic, and also as a way to share all our solutions, we have formatted the coefficients as C code (see supplementary material) to evaluate the approximations in 24-bit (mantissa) single, 53-bit double, 64-bit long double, and 113-bit quadruple precision, and to compare it to the standard library erf\\erf function. As our “standard” single- and double-precision approximations we have chosen those highlighted in Table [1](https://arxiv.org/html/2504.05068v1#S3.T1 "Table 1 ‣ 3. Computations ‣ Global approximations to the error functionof real argument for vectorized computation")
, and the rounding errors add up to leave us with about 22, 21 (single) and 51, 48 (double) bits of precision.

To measure the computational speed, we have written C code (see supplementary material) for serial as well as 4-way double- and 8-way single-precision vectorized calculation, such that the GCC \[[4](https://arxiv.org/html/2504.05068v1#bib.bib4)\
\] compiler we use can translate it into either scalar or vector instructions. For the AVX2/FMA instruction set, we get a quite well-optimized machine code (see supplementary material) where the scalar and vector instructions nearly parallel each other, and run it on an AMD 3950X 16-core processor running at 3.5 GHz clock frequency with SMT turned off, 16 identical jobs in parallel to load all the cores. Timing the repeated evaluation of a function f⁡(x)f(x) over 512 equally-spaced values of 0≤x<40\\leq x<4, for a total of about 2322^{32} function calls, is used to estimate the number of processor clock cycles for one function value including load/store, call/return, and looping operations.

We compare the speed of the standard C library \[[3](https://arxiv.org/html/2504.05068v1#bib.bib3)\
\] implementation of exp\\exp and erf\\erf functions against the serial and vectorized code of our approximations. We also use this occasion to share our own vectorized single- and double-precision implementation of the exponential function where not the traditional Chebyshev but the direct uniform approximation to the 2x2^{x} function for 12≤x≤12\\tfrac{1}{2}\\leq x\\leq\\tfrac{1}{2} is used.

Table 2. Measurements of computational speed.

| function | precision | method | vector | clock | speedup |     |
| --- | --- | --- | --- | --- | --- | --- |
|     |     |     | length | cycles | lib. | vec. |
| --- | --- | --- | --- | --- | --- | --- |
| exp\\exp | double | glibc \[[3](https://arxiv.org/html/2504.05068v1#bib.bib3)<br>\] | 1   | 45  |     |     |
| exp\\exp | double | ours | 1   | 16  | 2.8 |     |
| exp\\exp | double | ours | 4   | 18  |     | 3.6 |
| erf\\erf | double | glibc \[[3](https://arxiv.org/html/2504.05068v1#bib.bib3)<br>\] | 1   | 83  |     |     |
| erf\\erf | double | ours, Eq. [2.12](https://arxiv.org/html/2504.05068v1#S2.E12 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation") | 1   | 27  | 3.1 |     |
| erf\\erf | double | ours, Eq. [2.12](https://arxiv.org/html/2504.05068v1#S2.E12 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation") | 4   | 34  |     | 3.2 |
| erf\\erf, exp\\exp | double | ours, Eq. [2.7](https://arxiv.org/html/2504.05068v1#S2.E7 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation") | 1   | 43  |     |     |
| erf\\erf, exp\\exp | double | ours, Eq. [2.7](https://arxiv.org/html/2504.05068v1#S2.E7 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation") | 4   | 65  |     | 2.6 |
| exp\\exp | single | glibc \[[3](https://arxiv.org/html/2504.05068v1#bib.bib3)<br>\] | 1   | 19  |     |     |
| exp\\exp | single | ours | 1   | 10  | 1.9 |     |
| exp\\exp | single | ours | 8   | 12  |     | 6.7 |
| erf\\erf | single | glibc \[[3](https://arxiv.org/html/2504.05068v1#bib.bib3)<br>\] | 1   | 62  |     |     |
| erf\\erf | single | ours, Eq. [2.12](https://arxiv.org/html/2504.05068v1#S2.E12 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation") | 1   | 16  | 3.9 |     |
| erf\\erf | single | ours, Eq. [2.12](https://arxiv.org/html/2504.05068v1#S2.E12 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation") | 8   | 20  |     | 6.4 |
| erf\\erf, exp\\exp | single | ours, Eq. [2.7](https://arxiv.org/html/2504.05068v1#S2.E7 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation") | 1   | 26  |     |     |
| erf\\erf, exp\\exp | single | ours, Eq. [2.7](https://arxiv.org/html/2504.05068v1#S2.E7 "In 2. Approximations ‣ Global approximations to the error functionof real argument for vectorized computation") | 8   | 38  |     | 5.5 |

Table [2](https://arxiv.org/html/2504.05068v1#S3.T2 "Table 2 ‣ 3. Computations ‣ Global approximations to the error functionof real argument for vectorized computation")
 shows our measurements, we see a not-so-unexpected speedup against the standard library, and a rather good vectorization speedup — less than ideal because, among other things, the processor shows greater superscalar capabilities when fed with a scalar instruction stream.

4\. Conclusions
---------------

We have found two new kinds of global closed-form approximations to the error function, and determined their coefficients and accuracy. The number of terms needed to reach an accuracy of up to 128 bits is rather small. Tests of a practical implementation using the (24-bit) single- and (53-bit) double-precision arithmetic show a speed high enough to outperform on average a standard library routine in serial computation, whereas the code is straightforward to vectorize and shows then a close-to-ideal performance.

References
----------

*   \[1\] S. F. Boys, _Electronic wave functions - i. a general method of calculation for the stationary states of any molecular system_, Proc. R. Soc. A 200 (1950), 542.
*   \[2\] W. J. Cody, _Rational chebyshev approximations for the error function_, Math. Comp. 23 (1969), 631–637.
*   \[3\] Free Software Foundation, _The gnu c library, version 2.21_, [https://www.gnu.org/software/libc/](https://www.gnu.org/software/libc/)
    , 2015.
*   \[4\] by same author, _Gcc, the gnu compiler collection, version 13.2.0_, [https://gcc.gnu.org/](https://gcc.gnu.org/)
    , 2023.
*   \[5\] P. M. W. Gill, R. D. Adamson, and J. A. Pople, _Coulomb-attenuated exchange energy density functionals_, Mol. Phys. 88 (1996), 1005.
*   \[6\] J.W.L. Glaisher, _Xxxii. on a class of definite integrals_, London, Edinburgh Dublin Philos. Mag. J. Sci. 42 (1871), 294–302.
*   \[7\] Roger G. Hart, _A formula for the approximation of definite integrals of the normal distribution function_, MTAC 11 (1957), 265.
*   \[8\] by same author, _A close approximation related to the error function_, Math. Comput. 20 (1966), 600–602.
*   \[9\] Cecil Hastings, Jr., _Approximations for digital computers_, Princeton University Press, Princeton, N. J., 1955, Assisted by Jeanne T. Hayward and James P. Wong, Jr.
*   \[10\] K. B. Oldham, _Approximations for the x​exp⁡x2​erfc​xx\\exp x^{2}\\erfc x function_, Math. Comp. 22 (1968), 454–454.
*   \[11\] M. M. Shepherd and J. G. Laframboise, _Chebyshev approximation of (1+2​x)​exp​(x2)​erfc​x(1+2x)\\mathrm{exp}(x^{2})\\mathrm{erfc}x in 0≤x<∞0\\leq x<\\infty_, Math. Comp. 36 (1981), 249–253.

Experimental support, please [view the build logs](https://arxiv.org/html/2504.05068v1/__stdout.txt)
 for errors. Generated by [L A T E xml ![[LOGO]](<Base64-Image-Removed>)](https://math.nist.gov/~BMiller/LaTeXML/)
  .

Instructions for reporting errors
---------------------------------

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

*   Click the "Report Issue" ( ) button, located in the page header.

**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues)
. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML)
, and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues)
.

We gratefully acknowledge support from our **major funders**, [**member institutions**](https://info.arxiv.org/about/ourmembers.html)
, , and all contributors.

[About](https://info.arxiv.org/about)
 · [Help](https://info.arxiv.org/help)
 · [Contact](https://info.arxiv.org/help/contact.html)
 · [Subscribe](https://info.arxiv.org/help/subscribe)
 · [Copyright](https://info.arxiv.org/help/license/index.html)
 · [Privacy](https://info.arxiv.org/help/policies/privacy_policy.html)
 · [Accessibility](https://info.arxiv.org/help/web_accessibility.html)
 · [Operational Status (opens in new tab)](https://status.arxiv.org/)

Major funding support from

 [![Simons Foundation](https://arxiv.org/static/base/1.0.1/images/funders/simons-foundation.png)](https://www.simonsfoundation.org/)
[![Simons Foundation International](https://arxiv.org/static/base/1.0.1/images/funders/simons-foundation-international.png)](https://www.sfi.org.bm/)
[![Schmidt Sciences](https://arxiv.org/static/base/1.0.1/images/funders/schmidt-sciences.png)](https://www.schmidtsciences.org/)

[](javascript:toggleReadingMode(); "Disable reading mode, show header and footer")