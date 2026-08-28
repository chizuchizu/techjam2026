[![John Hauser](https://www.jhauser.us/arithmetic/byline.gif)](https://www.jhauser.us/)

* * *

Berkeley SoftFloat
==================

_Releases 3 through 3c of Berkeley SoftFloat contain bugs in the square root functions that may be of concern for some uses. Those bugs are believed to be repaired in Release 3d and later, as explained below._

Berkeley SoftFloat is a free, high-quality software implementation of binary floating-point that conforms to the IEEE Standard for Floating-Point Arithmetic. SoftFloat is completely faithful to the IEEE Standard, while at the same time being relatively fast. All functions dictated by the original 1985 version of the standard are supported except for conversions to and from decimal. The latest release of SoftFloat implements five floating-point formats: 16-bit half-precision, 32-bit single-precision, 64-bit double-precision, 80-bit double-extended-precision, and 128-bit quadruple-precision. All required rounding modes, exception flags, and special values are supported. Fused multiply-add is also implemented for all formats except 80-bit double-extended-precision.

SoftFloat is distributed in the form of ISO/ANSI C source code and should be compilable with almost any ISO-compliant C compiler. Using the GNU C Compiler (`gcc`), the package has been compiled and tested for several platforms. Target-specific code is provided for various Intel x86 and ARM processors. Other machines can be targeted using these as examples.

Since Release 3, SoftFloat has depended on the existence of a 64-bit integer type in C. If the C compiler used to compile SoftFloat does not support 64-bit integers, it is still possible to use the older Release 2c to implement the two most common formats, 32-bit single-precision and 64-bit double-precision, but not the other formats.

Release 3e
----------

Release 3 was a complete rewrite of SoftFloat, funded by the University of California, Berkeley. Compared to earlier releases, Release 3 added functions for converting to and from unsigned integers, functions for fused multiply-add, and better algorithms for division, remainder, and square root. Unlike past versions, Release 3 and later have a U.C. Berkeley open-source license (specified in the documentation).

Release 3b of Berkeley SoftFloat added support for the 16-bit half-precision format, and Release 3c added optional support for a rarely used rounding mode, _round to odd_, also known as _jamming_.

Release 3d (2017 August) fixed bugs that were found in the square root functions for the 64-bit, 80-bit, and 128-bit floating-point formats. (Thanks to Alexei Sibidanov at the University of Victoria for reporting an incorrect result.) The bugs affected all prior Release-3 versions of SoftFloat. For 64-bit double-precision, the flaw in the square root function was of very minor impact, causing a 1-ulp error (1 unit in the last place) a few times out of a billion. The bugs in the 80-bit and 128-bit square root functions were more serious. Although incorrect results again occurred only a few times out of a billion, when they did occur a large portion of the less-significant bits could be wrong.

The latest version of SoftFloat is Release 3e (2018 January). This release incorporates a number of minor fixes and improvements, including new specialization code to model the floating-point of ARM processors. More information about this release is in the following files from the SoftFloat package:

|     |     |
| --- | --- |
| [![--](https://www.jhauser.us/arithmetic/icon-bullet.gif)](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat.html) | [`SoftFloat.html`](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat.html)<br> - _Berkeley SoftFloat Release 3e: Library Interface_. This document includes a section on the many differences between Release 3 and the earlier Release 2. |
| [![--](https://www.jhauser.us/arithmetic/icon-bullet.gif)](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat-source.html) | [`SoftFloat-source.html`](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat-source.html)<br> - _Berkeley SoftFloat Release 3e: Source Documentation_. |
| [![--](https://www.jhauser.us/arithmetic/icon-bullet.gif)](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat-history.html) | [`SoftFloat-history.html`](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat-history.html)<br> - _History of Berkeley SoftFloat, to Release 3e_. |

The following archive contains all source code and documentation for Release 3e:

|     |     |
| --- | --- |
| [![->](https://www.jhauser.us/arithmetic/icon-file.gif)](https://www.jhauser.us/arithmetic/SoftFloat-3e.zip) | `zip` archive, [`SoftFloat-3e.zip`](https://www.jhauser.us/arithmetic/SoftFloat-3e.zip)<br> \[730 kB\]. |

The older Releases 3a through 3d (with known bugs) are still available here:

|     |     |
| --- | --- |
| [![->](https://www.jhauser.us/arithmetic/icon-file.gif)](https://www.jhauser.us/arithmetic/old/SoftFloat-3d.zip) | `zip` archive, [`SoftFloat-3d.zip`](https://www.jhauser.us/arithmetic/old/SoftFloat-3d.zip)<br> \[658 kB\]. |
| [![->](https://www.jhauser.us/arithmetic/icon-file.gif)](https://www.jhauser.us/arithmetic/old/SoftFloat-3c.zip) | `zip` archive, [`SoftFloat-3c.zip`](https://www.jhauser.us/arithmetic/old/SoftFloat-3c.zip)<br> \[657 kB\]. |
| [![->](https://www.jhauser.us/arithmetic/icon-file.gif)](https://www.jhauser.us/arithmetic/old/SoftFloat-3b.zip) | `zip` archive, [`SoftFloat-3b.zip`](https://www.jhauser.us/arithmetic/old/SoftFloat-3b.zip)<br> \[655 kB\]. |
| [![->](https://www.jhauser.us/arithmetic/icon-file.gif)](https://www.jhauser.us/arithmetic/old/SoftFloat-3a.zip) | `zip` archive, [`SoftFloat-3a.zip`](https://www.jhauser.us/arithmetic/old/SoftFloat-3a.zip)<br> \[563 kB\]. |

Release 2c
----------

For those who may need it, SoftFloat versions preceding Release 3 have been updated to Release 2c (2015 January). The only changes in this release compared to earlier releases 2a and 2b are these:

*   Some bugs are fixed that affected correct compilation for some 64-bit processors only.
*   The documentation has been improved in minor ways.
*   The restrictions on legal use have been further clarified (not applicable to Release 3 or later).

If you have been successfully using Release 2a or 2b on a 32-bit processor (or compiled as though for a 32-bit processor), you probably do not need to download this release.

|     |     |
| --- | --- |
| [![->](https://www.jhauser.us/arithmetic/icon-file.gif)](https://www.jhauser.us/arithmetic/SoftFloat-2c.zip) | `zip` archive, [`SoftFloat-2c.zip`](https://www.jhauser.us/arithmetic/SoftFloat-2c.zip)<br> \[108 kB\]. |

Testing SoftFloat
-----------------

Once compiled, SoftFloat can be tested using the `testsoftfloat` program of Berkeley TestFloat:

|     |     |
| --- | --- |
| [![[]](https://www.jhauser.us/arithmetic/icon-link.gif)](https://www.jhauser.us/arithmetic/TestFloat.html) | **[Berkeley TestFloat](https://www.jhauser.us/arithmetic/TestFloat.html)<br>** is a small collection of programs for testing that an implementation of binary floating-point conforms to the IEEE Standard for Floating-Point Arithmetic. |

SoftFloat speed
---------------

The following table shows the speeds of some of SoftFloat’s functions on machines of different strengths. For this table, SoftFloat was compiled using the GNU C Compiler with only ordinary optimization enabled (`gcc` `-O2`) together with the GCC-specific optimizations in the provided source code. Speeds are reported in Mop/s (millions of floating-point operations per second). Function times are affected by operand values and cache effects, among other factors, so your results may vary.

|     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | speed (Mop/s) |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
|     | 32-bit single |     |     |     | 64-bit double |     |     |     | 80-bit  <br>double-extended |     |     |     | 128-bit  <br>quadruple |     |     |
|     | add | mul | div | add | mul | div | add | mul | div | add | mul | div |
| 1-GHz ARM Cortex-A8 (32-bit),  <br>32-kB I/D caches, Linux, GCC 4.6.3 | 8.09 | 10.0 | 7.04 | 5.13 | 5.30 | 4.24 | 5.22 | 5.30 | 2.05 | 3.51 | 2.32 | 1.13 |
| 3.4-GHz Intel Core i7-6700 (64-bit, with  <br>4.0-GHz turbo boost), Linux, GCC 4.9.2 | 82.3 | 109 | 62.5 | 75.7 | 119 | 49.4 | 70.0 | 89.9 | 36.1 | 56.9 | 56.4 | 26.7 |

Frequently asked questions
--------------------------

A couple of frequently asked questions about SoftFloat are addressed in this document:

|     |     |
| --- | --- |
| [![--](https://www.jhauser.us/arithmetic/icon-bullet.gif)](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat-FAQ.html) | [`SoftFloat-FAQ.html`](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat-FAQ.html)<br> - _Berkeley SoftFloat: Frequently Asked Questions_. |

Related sites
-------------

|     |     |
| --- | --- |
| [![[]](https://www.jhauser.us/arithmetic/icon-link.gif)](https://www.netlib.org/fp/) | The [fp](https://www.netlib.org/fp/)<br> (_f_loating-_p_oint) directory of the [Netlib Repository](https://www.netlib.org/)<br>. Includes C subroutines for converting to and from decimal textual formats (e.g., strtod). |
| [![[]](https://www.jhauser.us/arithmetic/icon-link.gif)](https://www.netlib.org/fdlibm/) | The **[Freely Distributable LIBM](https://www.netlib.org/fdlibm/)<br>** (fdlibm), an implementation of the C math library (sin, exp, log, etc.), also at the [Netlib Repository](https://www.netlib.org/)<br>. |
| [![[]](https://www.jhauser.us/arithmetic/icon-link.gif)](https://docs.oracle.com/cd/E19422-01/819-3693/) | Sun Microsystems’ **[_Numerical Computation Guide_](https://docs.oracle.com/cd/E19422-01/819-3693/)<br>**, which covers the IEEE Standard in detail. (Sun Microsystems was acquired by Oracle in 2010.) |
| [![[]](https://www.jhauser.us/arithmetic/icon-link.gif)](https://www.jhauser.us/publications/1996_Hauser_FloatingPointExceptions.html) | My article “**[Handling floating-point exceptions in numeric programs](https://www.jhauser.us/publications/1996_Hauser_FloatingPointExceptions.html)<br>**,” explaining how the exception handling features of the IEEE Standard can be applied to real programs. |

Credit and contacts
-------------------

Berkeley SoftFloat was written by me, [John R. Hauser](https://www.jhauser.us/)
. Funding for the development of SoftFloat Release 3 and later was provided indirectly by portions of grants to U.C. Berkeley from Microsoft, Intel, DARPA, Nokia, NVIDIA, Oracle, Samsung, and Google, and by a portion of a U.C. Discovery Grant. The SoftFloat documentation has more details.

Bugs in SoftFloat and other comments can be reported to me at `j``h``@``jhauser``.``u``s`.

  

* * *

[John Hauser](https://www.jhauser.us/)
, 2024 October 18