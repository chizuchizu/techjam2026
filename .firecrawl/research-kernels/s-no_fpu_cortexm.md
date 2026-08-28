[Skip to content](https://www.systemonchips.com/options-for-floating-point-math-on-cortex-m-without-fpus/#main)

🚀 New: [ARM Technical Advisor](https://www.systemonchips.com/arm-technical-advisor/)
 - Get instant expert help for free!

[System on Chips](https://www.systemonchips.com/)

*   [Technical Advisor](https://www.systemonchips.com/arm-technical-advisor/)
    

Search

[System on Chips](https://www.systemonchips.com/)

Toggle Menu

Options for Floating Point Math on Cortex M Without FPUs
========================================================

By[Neil Salmon](https://www.systemonchips.com/author/neil-salmon/)
 Updated onSeptember 27, 2023

For Cortex M chips without a dedicated floating point unit (FPU), performing floating point math operations efficiently can be challenging. However, with the right software libraries and techniques, it is possible to do floating point math on Cortex M CPUs lacking an FPU. This article explores the options and tradeoffs for implementing floating point math on FPU-less Cortex M microcontrollers.

Why Floating Point Math on Cortex M?
------------------------------------

While Cortex M chips are intended for low-cost and low-power embedded applications, increasingly these applications require some floating point math capabilities. Examples include:

*   Digital signal processing algorithms involving filters, transforms, etc.
*   Embedded machine learning using neural networks
*   Sensor data processing from pressure, temperature, accelerometer inputs
*   Motor control applications with PI and PID loops

Rather than using a more expensive Cortex M chip with dedicated FPU, or moving to a different architecture, developers may want to enable floating point math on their existing Cortex M design. The key reasons are to minimize cost and maximize power efficiency.

Challenges of Software Floating Point
-------------------------------------

The Cortex M0/M0+/M1 chips lack an FPU, so floating point operations need to be handled in software. This brings several challenges:

*   Floating point math is complex, requiring algorithms for addition, subtraction, multiplication, division, square root, etc.
*   Software libraries take up flash and RAM on the already memory constrained Cortex M chips.
*   Execution speed is much slower than dedicated hardware FPU.
*   Precision is limited due to 32-bit single precision in software vs 64-bit double precision in hardware FPU.
*   Special handling needed for overflow, underflow, denormalized numbers, etc.
*   Consistency and correctness of results across toolchains, compilers, and libraries.

While performance and precision are limited, with careful coding and optimization, useful floating point math is achievable on Cortex M0/M0+/M1.

Software Floating Point Libraries
---------------------------------

Several open source libraries are available that provide floating point math functions for Cortex M chips without FPU:

### CMSIS Software FP

CMSIS Software FP is provided by ARM as part of the Cortex Microcontroller Software Interface Standard (CMSIS). It provides a common software interface for Cortex M cores. The software FP implementation includes C functions for addition, subtraction, multiplication, division, square root, trigonometric, exponential, logarithmic and other math functions. Key features:

*   ANSI/IEEE 754 compliant 32-bit single precision
*   Pure C implementation, no asm or compiler intrinsics needed
*   Hand optimized algorithms using integer math
*   Configurable to tradeoff performance vs precision
*   MIT open source license

CMSIS software FP is a good choice when adherence to a standard math library API is desired. It is included with many IDEs and toolchains.

### Berkeley SoftFloat

Berkeley SoftFloat is an open source floating point library targeted at systems without an FPU. It implements 32-bit and 64-bit floating point per the IEEE 754 standard. Key features include:

*   Pure C implementation with optimizing options
*   Thread safe, re-entrant code
*   Portable across many architectures
*   Separate libraries for single, double precision
*   BSD open source license

For applications requiring portability across hardware platforms, including wider 64-bit types, SoftFloat is a good choice. The separate libraries allow optimized single precision math.

### Other Options

In addition to CMSIS and SoftFloat, some other floating point software libraries include:

*   cephes – collection of math functions in C
*   FDLIBM – faithfully rounded math library
*   libm – standard C math library, adapted for embedded
*   MPLA – multiprecision floating point library
*   MuLib – microcontroller library for high precision math

Developers can evaluate their specific application requirements when selecting among these software floating point libraries.

Floating Point Code Optimizations
---------------------------------

When using software floating point libraries, developers can employ various optimizations to improve performance on Cortex M CPUs:

### Precomputation

Compute costly math results once upfront, cache and reuse the results rather than recompute. Trading off some RAM for faster execution.

### Approximation

Use polynomial or linear approximations for transcendental functions like sine, cosine, logarithms. Gives accuracy vs performance tradeoff.

### Loop Unrolling

Unroll small fixed loops to reduce overhead of branches and loop counter logic in innermost math operations.

### Assembly Optimize

Hand optimize key algorithms in assembly language, leveraging available registers and data types for massive speedups.

### Intrinsic Functions

Use compiler intrinsic functions to generate SIMD or other specialized instructions where supported, avoiding library function call overhead.

### Reduce Precision

Use lower precision data types like 16-bit float or fixed point where possible to gain performance at the cost of precision.

Profiling tools can help identify optimization opportunities and quantify performance gains from various techniques.

Leveraging Fixed Point Math
---------------------------

While floating point is necessary for some applications, fixed point math may meet requirements in other cases. With fixed point, computations are performed on integers rather than floats. Benefits of fixed point math include:

*   Higher precision for a given word length
*   No special handling of edge cases like denormals, underflow, etc.
*   Slightly faster computation than floats
*   Deterministic behavior

The limitations are fixed point’s lower dynamic range vs floats, and needing to scaling values to preserve precision. Fixed point math libraries like QMath provide optimized fixed point operations.

Leveraging Low Power Math Accelerators
--------------------------------------

Some Cortex M based microcontrollers include integrated math accelerators to offload intensive floating point or fixed point math sequences. These help reduce software overhead and memory usage. Some examples include:

*   STM32L4/L4+ – CORDIC accelerator for trig, hyperbolic, log, exp functions
*   STM32F7 – FMAC accelerator for 32-bit fixed point multiply-accumulates
*   NXP Kinetis – FlexISP digital signal processor
*   TI MSP432 – Low energy accelerator (LEA) for FIR filters

When available, math accelerators can provide 10-100x speedups on accelerated algorithms with minimal software and power overhead.

Leveraging External Math Chips
------------------------------

For more intensive floating point requirements, an external math coprocessor chip can be added to offload the Cortex M CPU. Options include:

*   FPGA – Custom floating point logic implemented in FPGA fabric
*   GPU – Graphical processing unit efficient at math operations
*   DSP – Digital signal processor optimized for math algorithms
*   SoC – Higher end Cortex-A/R/M cores with built-in FPU

The tradeoff is increased cost, complexity and power versus maximum performance on complex algorithms. External chips best for high performance needs.

Conclusion
----------

For Cortex M microcontrollers without an FPU, efficient floating point math is achievable using software libraries, code optimizations, fixed point math, hardware accelerators, and external coprocessors. Performance and precision requirements determine best approach.

Software libraries like CMSIS or SoftFloat provide portable floating point using pure C. Code optimizations and fixed point math can improve software performance. Hardware accelerators and coprocessors maximize performance for complex floating point operations.

With careful coding and the techniques outlined, useful floating point math is realizable on even low-cost FPU-less Cortex M chips.

Post navigation
---------------

[Previous Previous\
\
When to Use Hardware vs Software Floating Point with Arm Cortex M?](https://www.systemonchips.com/when-to-use-hardware-vs-software-floating-point-with-arm-cortex-m/)

[NextContinue\
\
Soft Float vs Hardware Floating Point Tradeoffs on Microcontrollers](https://www.systemonchips.com/soft-float-vs-hardware-floating-point-tradeoffs-on-microcontrollers/)

Similar Posts
-------------

[Arm](https://www.systemonchips.com/category/arm/)

### [ARM Cortex M0 Programming in C](https://www.systemonchips.com/arm-cortex-m0-programming-in-c/)

By[Elijah Erickson](https://www.systemonchips.com/author/elijan-erickson/)
 September 7, 2023

The ARM Cortex-M0 is a 32-bit processor designed for low-cost and low-power embedded applications. With its simple, compact design, the Cortex-M0 is well-suited for basic microcontroller applications that don’t require the performance of more advanced ARM cores. Programming the Cortex-M0 in C provides a good balance of performance, portability, and ease of development. Introduction to…

[Read More ARM Cortex M0 Programming in CContinue](https://www.systemonchips.com/arm-cortex-m0-programming-in-c/)

[Arm](https://www.systemonchips.com/category/arm/)

### [Which Stack Is Used Coming Out of Reset In ARM Cortex-M, MSP or PSP?](https://www.systemonchips.com/which-stack-is-used-coming-out-of-reset-in-arm-cortex-m-msp-or-psp/)

By[Neil Salmon](https://www.systemonchips.com/author/neil-salmon/)
 September 16, 2023

When an ARM Cortex-M processor comes out of reset, it will start executing code from the vector table located at address 0x00000000. The processor expects the first entry in the vector table to be the initial stack pointer value. This initial stack pointer value determines whether the Main Stack Pointer (MSP) or Process Stack Pointer…

[Read More Which Stack Is Used Coming Out of Reset In ARM Cortex-M, MSP or PSP?Continue](https://www.systemonchips.com/which-stack-is-used-coming-out-of-reset-in-arm-cortex-m-msp-or-psp/)

[Arm](https://www.systemonchips.com/category/arm/)

### [ARM Cortex M Registers](https://www.systemonchips.com/arm-cortex-m-registers/)

By[Graham Kruk](https://www.systemonchips.com/author/graham-kruk/)
 September 8, 2023

The ARM Cortex-M is a group of 32-bit RISC ARM processor cores licensed by Arm Holdings. The Cortex-M cores are designed for microcontroller use, and consist of the Cortex-M0, Cortex-M0+, Cortex-M1, Cortex-M3, Cortex-M4, Cortex-M7, Cortex-M23, Cortex-M33, Cortex-M35P, Cortex-M55 and Cortex-M85/M85+ cores. They aim to provide a low-cost platform while having better performance than classic 8-bit…

[Read More ARM Cortex M RegistersContinue](https://www.systemonchips.com/arm-cortex-m-registers/)

[Arm](https://www.systemonchips.com/category/arm/)

### [ARM Cortex M4 Registers](https://www.systemonchips.com/arm-cortex-m4-registers/)

By[Graham Kruk](https://www.systemonchips.com/author/graham-kruk/)
 September 10, 2023

The ARM Cortex-M4 is a 32-bit processor core designed for embedded applications requiring high performance and low power consumption. It implements the ARMv7-M architecture and includes features like digital signal processing (DSP) instructions, single cycle multiply and divide operations, memory protection unit (MPU), and nested vectored interrupt controller (NVIC). Like all ARM Cortex-M cores, the…

[Read More ARM Cortex M4 RegistersContinue](https://www.systemonchips.com/arm-cortex-m4-registers/)

[Arm](https://www.systemonchips.com/category/arm/)

### [What are the different debug interfaces that are available on the Cortex-M processor?](https://www.systemonchips.com/what-are-the-different-debug-interfaces-that-are-available-on-the-cortex-m-processor/)

By[Scott Allen](https://www.systemonchips.com/author/scott-allen/)
 September 11, 2023

The Cortex-M processor family offers several debug interfaces that provide access to the core’s internal registers, memory, and peripherals for debugging and software development. The main debug interfaces available are SWD, JTAG, ETB, and DAP. Choosing the right interface depends on the capabilities required and constraints like cost and PCB area. SWD Interface The Serial…

[Read More What are the different debug interfaces that are available on the Cortex-M processor?Continue](https://www.systemonchips.com/what-are-the-different-debug-interfaces-that-are-available-on-the-cortex-m-processor/)

[Arm](https://www.systemonchips.com/category/arm/)

### [What is ARM Cortex-M35P?](https://www.systemonchips.com/what-is-arm-cortex-m35p/)

By[Elijah Erickson](https://www.systemonchips.com/author/elijan-erickson/)
 September 7, 2023

The ARM Cortex-M35P is a 32-bit processor core designed for microcontroller applications by ARM Holdings. It is part of the Cortex-M series of cores, which target low-cost and low-power embedded systems. Overview The Cortex-M35P core is designed for embedded applications requiring high performance and power efficiency. It provides higher performance than the previous Cortex-M33 core…

[Read More What is ARM Cortex-M35P?Continue](https://www.systemonchips.com/what-is-arm-cortex-m35p/)

### Leave a Reply [Cancel reply](https://www.systemonchips.com/options-for-floating-point-math-on-cortex-m-without-fpus/#respond)

Your email address will not be published. Required fields are marked \*

Comment \*

Name \*

Email \*

Website

 Save my name, email, and website in this browser for the next time I comment.

  
  

© 2026 System on Chips. All rights reserved.

*   [About](https://www.systemonchips.com/about/)
    
*   [Contact Us](https://www.systemonchips.com/contact-us/)
    
*   [Privacy Policy](https://www.systemonchips.com/privacy-policy/)
    

*   [Technical Advisor](https://www.systemonchips.com/arm-technical-advisor/)
    

Toggle Menu Close

Search for:  

Search