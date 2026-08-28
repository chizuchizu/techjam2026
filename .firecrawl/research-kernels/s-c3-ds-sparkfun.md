

ESP32­C3 Family
Datasheet
Ultra­Low­Power SoC with RISC­V Single­Core CPU
Supporting IEEE 802.11b/g/n (2.4 GHz Wi­Fi) and Bluetooth 5 (LE)
Including:
ESP32-C3
ESP32-C3FN4
ESP32-C3FH4
Version 1.0
Espressif Systems
Copyright © 2021
www.espressif.com

Product Overview
ESP32-C3 family is an ultra-low-power and highly-integrated MCU-based SoC solution that supports 2.4 GHz
Wi-Fi and Bluetooth
®
Low Energy (Bluetooth LE). It has:
•A complete Wi-Fi subsystem that complies with
IEEE 802.11b/g/n protocol and supports Station
mode, SoftAP mode, SoftAP + Station mode,
and promiscuous mode
•A Bluetooth LE subsystem that supports features
of Bluetooth 5 and Bluetooth mesh
•State-of-the-art power and RF performance
•32-bit RISC-V single-core processor with a
four-stage pipeline that operates at up to 160
MHz
•400 KB of SRAM (16 KB for cache) and 384 KB
of ROM on the chip, and SPI, Dual SPI, Quad
SPI, and QPI interfaces that allow connection to
external flash
•Reliable security features ensured by
–Cryptographic hardware accelerators that
support AES-128/256, Hash, RSA, HMAC,
digital signature and secure boot
–Random number generator
–Permission control on accessing internal
memory, external memory, and peripherals
–External memory encryption and decryption
•Rich set of peripheral interfaces and GPIOs, ideal
for various scenarios and complex applications
Block Diagram
Cryptographic Hardware Acceleration
RTC
Wireless MAC and 
Baseband
Main CPU
JTAG
RTC memoryPMU
Wi-Fi MAC 
Wi-Fi 
baseband
RISC-V
32-bit
Microprocessor
ROM
Cache
SRAM
Peripherals and Sensors
RF
RF receiver
RF 
transmitter
Clock 
generator
SHARSA
AES
RNG
Espressif’s ESP32-C3 Wi-Fi + BLE SoC
HMACDigital signature
SPI I2S
Temperature 
sensor
LED PWMADC
TWAITimers
UART
GPIO
RMTGDMA
BLE 5.0 
link 
controller 
BLE 5.0 
baseband
XTS-AES-128 flash encryption
Switch
Balun
I2C
Embedded 
flash
USB 
Serial-7$*
Figure 1: Block Diagram of ESP32­C3
Espressif Systems1
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

Features
Wi­Fi
•IEEE 802.11 b/g/n-compliant
•Supports 20 MHz, 40 MHz bandwidth in 2.4
GHz band
•1T1R mode with data rate up to 150 Mbps
•Wi-Fi Multimedia (WMM)
•TX/RX A-MPDU, TX/RX A-MSDU
•Immediate Block ACK
•Fragmentation and defragmentation
•Transmit opportunity (TXOP)
•Automatic Beacon monitoring (hardware TSF)
•4 × virtual Wi-Fi interfaces
•Simultaneous support for Infrastructure BSS in
Station mode, SoftAP mode, Station + SoftAP
mode, and promiscuous mode
Note that when ESP32-C3 family scans in Station
mode, the SoftAP channel will change along with
the Station channel
•Antenna diversity
•802.11mc FTM
Bluetooth
•Bluetooth LE: Bluetooth 5, Bluetooth mesh
•Speed: 125 Kbps, 500 Kbps, 1 Mbps, 2 Mbps
•Advertising extensions
•Multiple advertisement sets
•Channel selection algorithm #2
CPU and Memory
•32-bit RISC-V single-core processor, up to 160
MHz
•384 KB ROM
•400 KB SRAM (16 KB for cache)
•8 KB SRAM in RTC
•Embedded flash (see details in Chapter1Family
Member Comparison)
•SPI, Dual SPI, Quad SPI, and QPI interfaces that
allow connection to multiple external flash
Advanced Peripheral Interfaces
•22 × programmable GPIOs
•Digital interfaces:
–3 × SPI
–2 × UART
–1 × I2C
–1 × I2S
–Remote control peripheral, with 2 transmit
channels and 2 receive channels
–LED PWM controller, with up to 6 channels
–Full-speed USB Serial/JTAG controller
–General DMA controller (GDMA), with 3
transmit channels and 3 receive channels
–1 × TWAI
®
controller (compatible with ISO
11898-1)
•Analog interfaces:
–2 × 12-bit SAR ADCs, up to 6 channels
–1 × temperature sensor
•Timers:
–2 × 54-bit general-purpose timers
–3 × watchdog timers
–1 × 52-bit system timer
Low Power Management
•Power Management Unit with four power modes
Security
•Secure boot
•Flash encryption
•4096-bit OTP, up to 1792 bits for users
•Cryptographic hardware acceleration:
–AES-128/256 (FIPS PUB 197)
Espressif Systems2
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

•Permission Control
•SHA Accelerator (FIPS PUB 180-4)
•RSA Accelerator
•Random Number Generator (RNG)
•HMAC
•Digital signature
Applications (A Non­exhaustive List)
With ultra-low power consumption, ESP32-C3 family is an ideal choice for IoT devices in the following
areas:
•SmartHome
–Light control
–Smart button
–Smart plug
–Indoor positioning
•IndustrialAutomation
–Industrial robot
–Mesh network
–Human machine interface (HMI)
–Industrial field bus
•HealthCare
–Health monitor
–Baby monitor
•ConsumerElectronics
–Smart watch and bracelet
–Over-the-top (OTT) devices
–Wi-Fi and Bluetooth speaker
–Logger toys and proximity sensing toys
•Smart Agriculture
–Smart greenhouse
–Smart irrigation
–Agriculture robot
•Retail and Catering
–POS machines
–Service robot
•Audio Device
–Internet music players
–Live streaming devices
–Internet radio players
•Generic Low-power IoT Sensor Hubs
•Generic Low-power IoT Data Loggers
Espressif Systems3
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

Contents
Contents
Product Overview1
Block Diagram1
Features2
Applications3
1 Family Member Comparison8
1.1  Family Nomenclature8
1.2  Comparison8
2 Pin Definition9
2.1  Pin Layout9
2.2  Pin Description9
2.3  Power Scheme11
2.4  Strapping Pins12
3 Functional Description14
3.1  CPU and Memory14
3.1.1 CPU14
3.1.2 Internal Memory14
3.1.3 External Flash14
3.1.4 Address Mapping Structure15
3.1.5 Cache15
3.2  System Clocks16
3.2.1 CPU Clock16
3.2.2 RTC Clock16
3.3  Analog Peripherals16
3.3.1 Analog-to-Digital Converter (ADC)16
3.3.2 Temperature Sensor16
3.4  Digital Peripherals16
3.4.1 General Purpose Input / Output Interface (GPIO)16
3.4.2 Serial Peripheral Interface (SPI)18
3.4.3 Universal Asynchronous Receiver Transmitter (UART)19
3.4.4 I2C Interface19
3.4.5 I2S Interface19
3.4.6 Remote Control Peripheral19
3.4.7 LED PWM Controller20
3.4.8 General DMA Controller20
3.4.9 USB Serial/JTAG Controller20
3.4.10 TWAI
®
Controller20
3.5  Radio and Wi-Fi21
3.5.1 2.4 GHz Receiver21
3.5.2 2.4 GHz Transmitter21
3.5.3 Clock Generator21
Espressif Systems4
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

Contents
3.5.4 Wi-Fi Radio and Baseband21
3.5.5 Wi-Fi MAC22
3.5.6 Networking Features22
3.6  Bluetooth LE22
3.6.1 Bluetooth LE Radio and PHY22
3.6.2 Bluetooth LE Link Layer Controller23
3.7  Low Power Management23
3.8  Timers23
3.8.1 General Purpose Timers23
3.8.2 System Timer24
3.8.3 Watchdog Timers24
3.9  Cryptographic Hardware Accelerators24
3.10 Physical Security Features25
3.11 Peripheral Pin Configurations25
4 Electrical Characteristics27
4.1  Absolute Maximum Ratings27
4.2  Recommended Operating Conditions27
4.3  VDD_SPI Output Characteristics27
4.4  DC Characteristics (3.3 V, 25 °C)28
4.5  ADC Characteristics28
4.6  Current Consumption28
4.7  Reliability29
4.8  Wi-Fi Radio30
4.8.1 Wi-Fi RF Transmitter (TX) Specifications30
4.8.2 Wi-Fi RF Receiver (RX) Specifications31
4.9  Bluetooth LE Radio32
4.9.1 Bluetooth LE RF Transmitter (TX) Specifications32
4.9.2 Bluetooth LE RF Receiver (RX) Specifications34
5 Package Information37
Revision History38
Solutions, Documentation and Legal Information40
Espressif Systems5
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

List of Tables
List of Tables
1   ESP32-C3 Family Member Comparison8
2   Pin Description9
3   Description of ESP32-C3 Family Power-up and Reset Timing Parameters12
4   Strapping Pins13
5   Parameter Descriptions of Setup and Hold Times for the Strapping Pin13
6   IO MUX Pin Functions17
7   Power-Up Glitches on Pins18
8   Connection Between ESP32-C3 Family and External Flash19
9   Peripheral Pin Configurations25
10  Absolute Maximum Ratings27
11  Recommended Operating Conditions27
12  VDD_SPI Output Characteristics27
13  DC Characteristics (3.3 V, 25 °C)28
14  ADC Characteristics28
15  Current Consumption Depending on RF Modes29
16  Current Consumption Depending on Work Modes29
17  Reliability Qualifications29
18  Wi-Fi Frequency30
19  TX Power with Spectral Mask and EVM Meeting 802.11 Standards30
20  TX EVM Test30
21  RX Sensitivity31
22  Maximum RX Level31
23  RX Adjacent Channel Rejection32
24  Bluetooth LE Frequency32
25  Transmitter Characteristics - Bluetooth LE 1 Mbps32
26  Transmitter Characteristics - Bluetooth LE 2 Mbps33
27  Transmitter Characteristics - Bluetooth LE 125 Kbps33
28  Transmitter Characteristics - Bluetooth LE 500 Kbps33
29  Receiver Characteristics - Bluetooth LE 1 Mbps34
30  Receiver Characteristics - Bluetooth LE 2 Mbps34
31  Receiver Characteristics - Bluetooth LE 125 Kbps35
32  Receiver Characteristics - Bluetooth LE 500 Kbps35
Espressif Systems6
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

List of Figures
List of Figures
1   Block Diagram of ESP32-C31
2   ESP32-C3 Family Nomenclature8
3   ESP32-C3 Pin Layout (Top View)9
4   ESP32-C3 Family Power Scheme11
5   ESP32-C3 Family Power-up and Reset Timing12
6   Setup and Hold Times for the Strapping Pin13
7   Address Mapping Structure15
8   QFN32 (5×5 mm) Package37
Espressif Systems7
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

1  Family Member Comparison
1.Family Member Comparison
1.1Family Nomenclature
ESP32-C3
F
H1
x
Chip family
Embedded flash
Flash temperature
H: High temperature
N: Normal temperature
Flash VL]H0%
Figure 2: ESP32­C3 Family Nomenclature
1.2Comparison
Table 1: ESP32­C3 Family Member Comparison
Ordering CodeEmbedded FlashAmbient Temperature (°C)Package (mm)
ESP32-C3—–40∼105QFN32 (5*5)
ESP32-C3FN44 MB–40∼85QFN32 (5*5)
ESP32-C3FH44 MB–40∼105QFN32 (5*5)
Espressif Systems8
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

2  Pin Definition
2.Pin Definition
2.1Pin Layout
1
2
3
4
5
6
7
817
18
19
20
21
22
23
24
9
10
11
12
13
141516
272829
30
31
32
26
25
GPIO10
GPIO9
GPIO8
MTDO
MTCK
VDD3P3_RTC
MTDI
MTMS
GPIO3
CHIP_EN
GPIO2
XTAL_32K_N
XTAL_32K_P
VDD3P3
VDD3P3
LNA_IN
VDDAVDDAXTAL_PXTAL_N
U0TXD
U0RXDGPIO19GPIO18
SPID
SPICLK
SPICS0
SPIWP
SPIHD
VDD_SPI
VDD3P3_CPU
ESP32-C3 Family
33 GND
SPIQ
Figure 3: ESP32­C3 Pin Layout (Top View)
2.2Pin Description
Table 2: Pin Description
NameNo.TypePower DomainFunction
LNA_IN1I/O—RF input and output
VDD3P32P
A
—Analog power supply
VDD3P33P
A
—Analog power supply
XTAL_32K_P4I/O/TVDD3P3_RTCGPIO0,  ADC1_CH0, XTAL_32K_P
Espressif Systems9
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

2  Pin Definition
NameNo.TypePower DomainFunction
XTAL_32K_N5I/O/TVDD3P3_RTCGPIO1,  ADC1_CH1, XTAL_32K_N
GPIO26I/O/TVDD3P3_RTCGPIO2,  ADC1_CH2, FSPIQ
CHIP_EN7IVDD3P3_RTC
High: on, enables the chip.
Low: off, the chip powers off.
Note: Do not leave the CHIP_EN pin floating.
GPIO38I/O/TVDD3P3_RTCGPIO3,  ADC1_CH3
MTMS9I/O/TVDD3P3_RTCGPIO4,  ADC1_CH4, FSPIHD,    MTMS
MTDI10I/O/TVDD3P3_RTCGPIO5,  ADC2_CH0, FSPIWP,    MTDI
VDD3P3_RTC11P
D
—Input power supply for RTC
MTCK12I/O/TVDD3P3_CPUGPIO6,FSPICLK,   MTCK
MTDO13I/O/TVDD3P3_CPUGPIO7,FSPID,MTDO
GPIO814I/O/TVDD3P3_CPUGPIO8
GPIO915I/O/TVDD3P3_CPUGPIO9
GPIO1016I/O/TVDD3P3_CPUGPIO10,FSPICS0
VDD3P3_CPU17P
D
—Input power supply for CPU IO
VDD_SPI18I/O/T/P
D
VDD3P3_CPUGPIO11, output power supply for flash
SPIHD19I/O/TVDD3P3_CPUGPIO12, SPIHD
SPIWP20I/O/TVDD3P3_CPUGPIO13, SPIWP
SPICS021I/O/TVDD3P3_CPUGPIO14, SPICS0
SPICLK22I/O/TVDD3P3_CPUGPIO15, SPICLK
SPID23I/O/TVDD3P3_CPUGPIO16, SPID
SPIQ24I/O/TVDD3P3_CPUGPIO17, SPIQ
GPIO1825I/O/TVDD3P3_CPUGPIO18, USB_D-
GPIO1926I/O/TVDD3P3_CPUGPIO19, USB_D+
U0RXD27I/O/TVDD3P3_CPUGPIO20, U0RXD
U0TXD28I/O/TVDD3P3_CPUGPIO21, U0TXD
XTAL_N29——External crystal output
XTAL_P30——External crystal input
VDDA31P
A
—Analog power supply
VDDA32P
A
—Analog power supply
GND33G—Ground
1
P
A
: analog power supply; P
D
: power supply for RTC IO; I: input; O: output; T: high impedance.
2
Ports of embedded flash correspond to pins of ESP32-C3FN4 and ESP32-C3FH4 as follows:
•CS# = SPICS0
•IO0/DI = SPID
•IO1/DO = SPIQ
•CLK = SPICLK
•IO2/WP# = SPIWP
•IO3/HOLD# = SPIHD
These pins are not recommended for other uses.
3
For the data port connection between ESP32-C3 family and external flash please refer to Section3.4.2
Serial Peripheral Interface (SPI).
4
The pin function in this table refers only to some fixed settings and do not cover all cases for signals that
can be input and output through the GPIO matrix. For more information on the GPIO matrix, please refer
to Chapter IO MUX and GPIO Matrix (GPIO, IO_MUX) in
ESP32-C3 Technical Reference Manual.
Espressif Systems10
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

2  Pin Definition
2.3Power Scheme
Digital pins of ESP32-C3 family are divided into three different power domains:
•VDD3P3_CPU
•VDD_SPI
•VDD3P3_RTC
VDD3P3_CPU is the input power supply for CPU.
VDD_SPI can be an input power supply or an output power supply.
VDD3P3_RTC is the input power supply for RTC analog domain and CPU.
The power scheme diagram is shown in Figure4.
VDD_SPI
Domain
RTC
Domain
CPU
Domain
LDOLDO
1.1 V1.1 V
VDD3P3_RTCVDD3P3_CPU
VDD_SPI
3.3 V
RTC IO
R
SPI
Figure 4: ESP32­C3 Family Power Scheme
When working as an output power supply, VDD_SPI can be powered by VDD3P3_CPU via R
SP I
(nominal 3.3 V).
VDD_SPI can be powered off via software to minimize the current leakage of flash in Deep-sleep mode.
Notes on CHIP_EN:
Figure
5shows the power-up and reset timing of ESP32-C3 family. Details about the parameters are listed in
Table3.
Espressif Systems11
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

2  Pin Definition
VDDA, 
VDD3P3,
VDD3P3_RTC,
VDD3P3_CPU
CHIP_EN
t
0
t
1
V
IL_nRST
2.8 V
Figure 5: ESP32­C3 Family Power­up and Reset Timing
Table 3: Description of ESP32­C3 Family Power­up and Reset Timing Parameters
Min
ParameterDescription
(μs)
t
0
Time between bringing up the VDDA, VDD3P3, VDD3P3_RTC, and
VDD3P3_CPU rails, and activating CHIP_EN
50
t
1
Duration of CHIP_EN signal level &lt; V
IL_nRST
(refer to its value in
Table
13) to reset the chip
50
2.4Strapping Pins
ESP32-C3 family has three strapping pins:
•GPIO2
•GPIO8
•GPIO9
Software can read the values of GPIO2, GPIO8 and GPIO9 from GPIO_STRAPPING field in GPIO_STRAP_REG
register. For register description, please refer to Section GPIO Matrix Register Summary in
ESP32-C3 Technical Reference Manual.
During the chip’s system reset, the latches of the strapping pins sample the voltage level as strapping bits of ”0”
or ”1”, and hold these bits until the chip is powered down or shut down.
Types of system reset include:
•power-on-reset
•RTC watchdog reset
•brownout reset
•analog super watchdog reset
•crystal clock glitch detection reset
By default, GPIO9 is connected to the internal pull-up resistor. If GPIO9 is not connected or connected to an
external high-impedance circuit, the latched bit value will be ”1”
To change the strapping bit values, you can apply the external pull-down/pull-up resistances, or use the host
MCU’s GPIOs to control the voltage level of these pins when powering on ESP32-C3 family.
Espressif Systems12
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

2  Pin Definition
After reset, the strapping pins work as normal-function pins.
Refer to Table4for a detailed boot-mode configuration of the strapping pins.
Table 4: Strapping Pins
Booting Mode
1
PinDefaultSPI BootDownload Boot
GPIO2N/A11
GPIO8N/ADon’t care1
GPIO9Internal pull-up10
Enabling/Disabling ROM Code Print During Booting
PinDefaultFunctionality
GPIO8N/A
When the value of eFuse field EFUSE_UART_PRINT_CONTROL is
0 (default), print is enabled and not controlled by GPIO8.
1, if GPIO8 is 0, print is enabled; if GPIO8 is 1, it is disabled.
2, if GPIO8 is 0, print is disabled; if GPIO8 is 1, it is enabled.
3, print is disabled and not controlled by GPIO8.
1
The strapping combination of GPIO8 = 0 and GPIO9 = 0 is invalid and will trigger unexpected behavior.
Figure6shows the setup and hold times for the strapping pin before and after the CHIP_EN signal goes high.
Details about the parameters are listed in Table5.
CHIP_EN
t
1
t
0
Strapping pin
V
IL_nRST
V
IH
Figure 6: Setup and Hold Times for the Strapping Pin
Table 5: Parameter Descriptions of Setup and Hold Times for the Strapping Pin
Min
ParameterDescription
(ms)
t
0
Setup time before CHIP_EN goes from low to high0
t
1
Hold time after CHIP_EN goes high3
Espressif Systems13
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
3.Functional Description
This chapter describes the functions of ESP32-C3 family.
3.1CPU and Memory
3.1.1CPU
ESP32-C3 family has a low-power 32-bit RISC-V single-core microprocessor with the following features:
•four-stage pipeline that supports a clock frequency of up to 160 MHz
•RV32IMC ISA
•32-bit multiplier and 32-bit divider
•up to 32 vectored interrupts at seven priority levels
•up to 8 hardware breakpoints/watchpoints
•up to 16 PMP regions
•JTAG for debugging
3.1.2Internal Memory
ESP32-C3’s internal memory includes:
•384 KB of ROM: for booting and core functions.
•400 KB of on­chip SRAM: for data and instructions. Of the 400 KB SRAM, 16 KB is configured for cache
•RTC FAST memory: 8 KB of SRAM that can be accessed by the main CPU. It can retain data in
Deep-sleep mode.
•4 Kbit of eFuse: 1792 bits are reserved for user data, such as encryption key and device ID.
•Embedded flash: See details in Chapter1Family Member Comparison.
3.1.3External Flash
ESP32-C3 family supports SPI, Dual SPI, Quad SPI, and QPI interfaces that allow connection to multiple external
flash.
CPU’s instruction memory space and read-only data memory space can map into external flash of ESP32-C3,
whose size can be 16 MB at most. ESP32-C3 family supports hardware encryption/decryption based on
XTS-AES to protect developers’ programs and data in flash.
Through high-speed caches, ESP32-C3 family can support at a time up to:
•8 MB of instruction memory space which can map into flash as individual blocks of 64 KB. 8-bit, 16-bit and
32-bit reads are supported.
•8 MB of data memory space which can map into flash as individual blocks of 64 KB. 8-bit, 16-bit and
32-bit reads are supported.
Note:
After ESP32-C3 family is initialized, software can customize the mapping of external flash into the CPU address space.
Espressif Systems14
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
3.1.4Address Mapping Structure
Figure 7: Address Mapping Structure
Note:
The memory space with gray background is not available for use.
3.1.5Cache
ESP32-C3 family has an eight-way set associative cache. This cache is read-only and has the following
features:
•size: 16 KB
•block size: 32 bytes
•pre-load function
•lock function
•critical word first and early restart
Espressif Systems15
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
3.2System Clocks
3.2.1CPU Clock
The CPU clock has three possible sources:
•external main crystal clock
•fast RC oscillator (typically about 17.5 MHz, and adjustable)
•PLL clock
The application can select the clock source from the three clocks above. The selected clock source drives the
CPU clock directly, or after division, depending on the application. Once the CPU is reset, the default clock
source would be the external main crystal clock divided by 2.
3.2.2RTC Clock
The RTC slow clock is used for RTC counter, RTC watchdog and low-power controller. It has three possible
sources:
•external low-speed (32 kHz) crystal clock
•internal slow RC oscillator (typically about 136 kHz, and adjustable)
•internal fast RC oscillator divided clock (derived from the fast RC oscillator divided by 256)
The RTC fast clock is used for RTC peripherals and sensor controllers. It has two possible sources:
•external main crystal clock divided by 2
•internal fast RC oscillator (typically about 17.5 MHz, and adjustable)
3.3Analog Peripherals
3.3.1Analog­to­Digital Converter (ADC)
ESP32-C3 family integrates two 12-bit SAR ADCs.
•ADC1 supports measurements on 5 channels, and is factory-calibrated.
•ADC2 supports measurements on 1 channel, and is not factory-calibrated.
For ADC characteristics, please refer to Table
14.
3.3.2Temperature Sensor
The temperature sensor generates a voltage that varies with temperature. The voltage is internally converted via
an ADC into a digital value.
The temperature sensor has a range of –40 °C to 125 °C. It is designed primarily to sense the temperature
changes inside the chip. The temperature value depends on factors like microcontroller clock frequency or I/O
load. Generally, the chip’s internal temperature is higher than the ambient temperature.
3.4Digital Peripherals
3.4.1General Purpose Input / Output Interface (GPIO)
ESP32-C3 family has 22 GPIO pins which can be assigned various functions by configuring corresponding
registers. Besides digital signals, some GPIOs can be also used for analog functions, such as ADC.
Espressif Systems16
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
All GPIOs have selectable internal pull-up or pull-down, or can be set to high impedance. When these GPIOs are
configured as an input, the input value can be read by software through the register. Input GPIOs can also be set
to generate edge-triggered or level-triggered CPU interrupts. All digital IO pins are bi-directional, non-inverting
and tristate, including input and output buffers with tristate control. These pins can be multiplexed with other
functions, such as the UART, SPI, etc. For low-power operations, the GPIOs can be set to holding state.
The IO MUX and the GPIO matrix are used to route signals from peripherals to GPIO pins. Together they provide
highly configurable I/O. Using GPIO Matrix, peripheral input signals can be configured from any IO pins while
peripheral output signals can be configured to any IO pins. Table6shows the IO MUX functions of each pin. For
more information about IO MUX and GPIO matrix, please refer to Chapter IO MUX and GPIO Matrix (GPIO,
IO_MUX) inESP32-C3 Technical Reference Manual.
Table 6: IO MUX Pin Functions
NameNo.Function 0Function 1Function 2ResetNotes
XTAL_32K_P4GPIO0GPIO0—0R
XTAL_32K_N5GPIO1GPIO1—0R
GPIO26GPIO2GPIO2FSPIQ1R
GPIO38GPIO3GPIO3—1R
MTMS9MTMSGPIO4FSPIHD1R
MTDI10MTDIGPIO5FSPIWP1R
MTCK12MTCKGPIO6FSPICLK1*G
MTDO13MTDOGPIO7FSPID1G
GPIO814GPIO8GPIO8—1—
GPIO915GPIO9GPIO9—3—
GPIO1016GPIO10GPIO10FSPICS01G
VDD_SPI18GPIO11GPIO11—0—
SPIHD19SPIHDGPIO12—3—
SPIWP20SPIWPGPIO13—3—
SPICS021SPICS0GPIO14—3—
SPICLK22SPICLKGPIO15—3—
SPID23SPIDGPIO16—3—
SPIQ24SPIQGPIO17—3—
GPIO1825GPIO18GPIO18—0USB, G
GPIO1926GPIO19GPIO19—0*USB
U0RXD27U0RXDGPIO20—3G
U0TXD28U0TXDGPIO21—4—
Reset
The default configuration of each pin after reset:
•0- input disabled, in high impedance state (IE = 0)
•1- input enabled, in high impedance state (IE = 1)
•2- input enabled, pull-down resistor enabled (IE = 1, WPD = 1)
•3- input enabled, pull-up resistor enabled (IE = 1, WPU = 1)
•4- output enabled, pull-up resistor enabled (OE = 1, WPU = 1)
Espressif Systems17
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
•0*- input disabled, pull-up resistor enabled (IE = 0, WPU = 0, USB_WPU = 1). See details in Notes
•1*- When the value of eFuse bit EFUSE_DIS_PAD_JTAG is
0, input enabled, pull-up resistor enabled (IE = 1, WPU = 1)
1, input enabled, in high impedance state (IE = 1)
We recommend pulling high or low GPIO pins in high impedance state to avoid unnecessary power
consumption. You may add pull-up and pull-down resistors in your PCB design referring to Table13, or enable
internal pull-up and pull-down resistors during software initialization.
Notes
•R- These pins have analog functions.
•USB- GPIO18 and GPIO19 are USB pins. The pull-up value of a USB pin is controlled by the pin’s pull-up
value together with USB pull-up value. If any of the two pull-up values is 1, the pin’s pull-up resistor will be
enabled. The pull-up resistors of USB pins are controlled by USB_SERIAL_JTAG_DP_PULLUP bit.
•G- These pins have glitches during power-up. See details in Table7.
Table 7: Power­Up Glitches on Pins
Typical Time Period
PinGlitch
1
(ns)
MTCKLow-level glitch5
MTDOLow-level glitch5
GPIO10Low-level glitch5
U0RXDLow-level glitch5
GPIO18Pull-up glitch50000
1
Low-level glitch: the pin is at a low level during the time period;
High-level glitch: the pin is at a high level during the time period;
Pull-up glitch: the pin is pulled up during the time period;
Pull-down glitch: the pin is pulled down during the time period.
3.4.2Serial Peripheral Interface (SPI)
ESP32-C3 family features three SPI interfaces (SPI0, SPI1, and SPI2). SPI0 and SPI1 can only be configured to
operate in SPI memory mode, while SPI2 can be configured to operate in both SPI memory and general-purpose
SPI modes.
•SPI Memory mode
In SPI memory mode, SPI0, SPI1 and SPI2 interface with external SPI memory. Data is transferred in bytes.
Up to four-line STR reads and writes are supported. The clock frequency is configurable to a maximum of
120 MHz in STR mode.
•SPI2 General­purpose SPI (GP­SPI) mode
When SPI2 acts as a general-purpose SPI, it can operate in master and slave modes. SPI2 supports
two-line full-duplex communication and single-/two-/four-line half-duplex communication in both master
and slave modes. The host’s clock frequency is configurable. Data is transferred in bytes. The clock
polarity (CPOL) and phase (CPHA) are also configurable. The SPI2 interface can connect to GDMA.
–In master mode, the clock frequency is 80 MHz at most, and the four modes of SPI transfer format are
supported.
Espressif Systems18
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
–In slave mode, the clock frequency is 60 MHz at most, and the four modes of SPI transfer format are
also supported.
In most cases, the data port connection between ESP32-C3 family and external flash is as follows:
Table 8: Connection Between ESP32­C3 Family and External Flash
External Flash Data Port
Chip Pin
SPI Single­Line ModeSPI Two­Line ModeSPI Four­Line Mode
SPID (SPID)DIIO0IO0
SPIQ (SPIQ)DOIO1IO1
SPIWP (SPIWP)WP#—IO2
SPIHD (SPIHD)HOLD#—IO3
3.4.3Universal Asynchronous Receiver Transmitter (UART)
ESP32-C3 family has two UART interfaces, i.e. UART0 and UART1, which support IrDA and asynchronous
communication (RS232 and RS485) at a speed of up to 5 Mbps. The UART controller provides hardware flow
control (CTS and RTS signals) and software flow control (XON and XOFF). Both UART interfaces connect to
GDMA via UHCI0, and can be accessed by the GDMA controller or directly by the CPU.
3.4.4I2C Interface
ESP32-C3 family has an I2C bus interface which is used for I2C master mode or slave mode, depending on the
user’s configuration. The I2C interface supports:
•standard mode (100 Kbit/s)
•fast mode (400 Kbit/s)
•up to 800 Kbit/s (constrained by SCL and SDA pull-up strength)
•7-bit and 10-bit addressing mode
•double addressing mode
•7-bit broadcast address
Users can configure instruction registers to control the I2C interface for more flexibility.
3.4.5I2S Interface
ESP32-C3 family includes a standard I2S interface. This interface can operate as a master or a slave in
full-duplex mode or half-duplex mode, and can be configured for 8-bit, 16-bit, 24-bit, or 32-bit serial
communication. BCK clock frequency, from 10 kHz up to 40 MHz, is supported.
The I2S interface supports TDM PCM, TDM MSB alignment, TDM standard, and PDM TX interface. It connects
to the GDMA controller.
3.4.6Remote Control Peripheral
The Remote Control Peripheral (RMT) supports two channels of infrared remote transmission and two channels
of infrared remote reception. By controlling pulse waveform through software, it supports various infrared and
other single wire protocols. All four channels share a 192 × 32-bit memory block to store transmit or receive
waveform.
Espressif Systems19
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
3.4.7LED PWM Controller
The LED PWM controller can generate independent digital waveform on six channels. The LED PWM
controller:
•can generate digital waveform with configurable periods and duty cycle. The accuracy of duty cycle can be
up to 18 bits.
•has multiple clock sources, including APB clock and external main crystal clock.
•can operate when the CPU is in Light-sleep mode.
•supports gradual increase or decrease of duty cycle, which is useful for the LED RGB color-gradient
generator.
3.4.8General DMA Controller
ESP32-C3 family has a general DMA controller (GDMA) with six independent channels, i.e. three transmit
channels and three receive channels. These six channels are shared by peripherals with DMA feature. The
GDMA controller implements a fixed-priority scheme among these channels.
The GDMA controller controls data transfer using linked lists. It allows peripheral-to-memory and
memory-to-memory data transfer at a high speed. All channels can access internal RAM.
Peripherals on ESP32-C3 family with DMA feature are SPI2, UHCI0, I2S, AES, SHA, and ADC.
3.4.9USB Serial/JTAG Controller
ESP32-C3 integrates a USB Serial/JTAG controller. This controller has the following features:
•USB 2.0 full speed compliant, capable of up to 12 Mbit/s transfer speed (Note that this controller does not
support the faster 480 Mbit/s high-speed transfer mode)
•CDC-ACM virtual serial port and JTAG adapter functionality
•programming embedded/external flash
•CPU debugging with compact JTAG instructions
•a full-speed USB PHY integrated in the chip
3.4.10TWAI
®
Controller
ESP32-C3 family has a TWAI
®
controller with the following features:
•compatible with ISO 11898-1 protocol
•standard frame format (11-bit ID) and extended frame format (29-bit ID)
•bit rates from 1 Kbit/s to 1 Mbit/s
•multiple modes of operation: Normal, Listen Only, and Self-Test (no acknowledgment required)
•64-byte receive FIFO
•acceptance filter (single and dual filter modes)
•error detection and handling: error counters, configurable error interrupt threshold, error code capture,
arbitration lost capture
Espressif Systems20
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
3.5Radio and Wi­Fi
The ESP32-C3 family radio consists of the following blocks:
•2.4 GHz receiver
•2.4 GHz transmitter
•bias and regulators
•balun and transmit-receive switch
•clock generator
3.5.12.4 GHz Receiver
The 2.4 GHz receiver demodulates the 2.4 GHz RF signal to quadrature baseband signals and converts them to
the digital domain with two high-resolution, high-speed ADCs. To adapt to varying signal channel conditions,
ESP32-C3 family integrates RF filters, Automatic Gain Control (AGC), DC offset cancelation circuits, and
baseband filters.
3.5.22.4 GHz Transmitter
The 2.4 GHz transmitter modulates the quadrature baseband signals to the 2.4 GHz RF signal, and drives the
antenna with a high-powered CMOS power amplifier. The use of digital calibration further improves the linearity of
the power amplifier.
Additional calibrations are integrated to cancel any radio imperfections, such as:
•carrier leakage
•I/Q amplitude/phase matching
•baseband nonlinearities
•RF nonlinearities
•antenna matching
These built-in calibration routines reduce the cost, time, and specialized equipment required for product
testing.
3.5.3Clock Generator
The clock generator produces quadrature clock signals of 2.4 GHz for both the receiver and the transmitter. All
components of the clock generator are integrated into the chip, including inductors, varactors, filters, regulators
and dividers.
The clock generator has built-in calibration and self-test circuits. Quadrature clock phases and phase noise are
optimized on chip with patented calibration algorithms which ensure the best performance of the receiver and the
transmitter.
3.5.4Wi­Fi Radio and Baseband
The ESP32-C3 family Wi-Fi radio and baseband support the following features:
•802.11b/g/n
•802.11n MCS0-7 that supports 20 MHz and 40 MHz bandwidth
•802.11n MCS32
Espressif Systems21
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
•802.11n 0.4μs guard interval
•data rate up to 150 Mbps
•RX STBC (single spatial stream)
•adjustable transmitting power
•antenna diversity
ESP32-C3 family supports antenna diversity with an external RF switch. This switch is controlled by one or
more GPIOs, and used to select the best antenna to minimize the effects of channel imperfections.
3.5.5Wi­Fi MAC
ESP32-C3 family implements the full 802.11 b/g/n Wi-Fi MAC protocol. It supports the Basic Service Set (BSS)
STA and SoftAP operations under the Distributed Control Function (DCF). Power management is handled
automatically with minimal host interaction to minimize the active duty period.
The ESP32-C3 family Wi-Fi MAC applies the following low-level protocol functions automatically:
•4 × virtual Wi-Fi interfaces
•infrastructure BSS in Station mode, SoftAP mode, Station + SoftAP mode, and promiscuous mode
•RTS protection, CTS protection, Immediate Block ACK
•fragmentation and defragmentation
•TX/RX A-MPDU, TX/RX A-MSDU
•transmit opportunity (TXOP)
•Wi-Fi multimedia (WMM)
•GCMP, CCMP, TKIP, WAPI, WEP, BIP, WPA2-PSK/WPA2-Enterprise, and WPA3-PSK/WPA3-Enterprise
•automatic beacon monitoring (hardware TSF)
•802.11mc FTM
3.5.6Networking Features
Espressif provides libraries for TCP/IP networking, ESP-WIFI-MESH networking, and other networking protocols
over Wi-Fi. TLS 1.0, 1.1 and 1.2 is also supported.
3.6Bluetooth LE
ESP32-C3 family includes a Bluetooth Low Energy subsystem that integrates a hardware link layer controller, an
RF/modem block and a feature-rich software protocol stack. It supports the core features of Bluetooth 5 and
Bluetooth mesh.
3.6.1Bluetooth LE Radio and PHY
Bluetooth Low Energy radio and PHY in ESP32-C3 family support:
•1 Mbps PHY
•2 Mbps PHY for higher data rates
•coded PHY for longer range (125 Kbps and 500 Kbps)
•listen before talk (LBT), implemented in hardware
Espressif Systems22
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
•antenna diversity with an external RF switch
This switch is controlled by one or more GPIOs, and used to select the best antenna to minimize the effects
of channel imperfections.
3.6.2Bluetooth LE Link Layer Controller
Bluetooth Low Energy Link Layer Controller in ESP32-C3 family support:
•LE advertising extensions, to enhance broadcasting capacity and broadcast more intelligent data
•multiple advertisement sets
•simultaneous advertising and scanning
•multiple connections in simultaneous central and peripheral roles
•adaptive frequency hopping and channel assessment
•LE channel selection algorithm #2
•connection parameter update
•high duty cycle non-connectable advertising
•LE privacy 1.2
•LE data packet length extension
•link layer extended scanner filter policies
•low duty cycle directed advertising
•link layer encryption
•LE Ping
3.7Low Power Management
With the use of advanced power-management technologies, ESP32-C3 family can switch between different
power modes.
•Active mode: CPU and chip radio are powered on. The chip can receive, transmit, or listen.
•Modem-sleep mode: The CPU is operational and the clock speed can be reduced. Wi-Fi base band,
Bluetooth LE base band, and radio are disabled, but Wi-Fi and Bluetooth LE connection can remain active.
•Light-sleep mode: The CPU is paused. Any wake-up events (MAC, host, RTC timer, or external interrupts)
will wake up the chip. Wi-Fi and Bluetooth LE connection can remain active.
•Deep-sleep mode: CPU and most peripherals are powered down. Only the RTC memory is powered on.
Wi-Fi connection data are stored in the RTC memory.
For power consumption in different power modes, please refer to Table
16.
3.8Timers
3.8.1General Purpose Timers
ESP32-C3 family is embedded with two 54-bit general-purpose timers, which are based on 16-bit prescalers
and 54-bit auto-reload-capable up/down-timers.
The timers’ features are summarized as follows:
Espressif Systems23
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
•a 16-bit clock prescaler, from 1 to 65536
•a 54-bit time-base counter programmable to be incrementing or decrementing
•able to read real-time value of the time-base counter
•halting and resuming the time-base counter
•programmable alarm generation
•level interrupt generation
3.8.2System Timer
ESP32-C3 family integrates a 52-bit system timer, which has two 52-bit counters and three comparators. The
system timer has the following features:
•counters with a fixed clock frequency of 16 MHz
•three types of independent interrupts generated according to alarm value
•two alarm modes: target mode and period mode
•52-bit target alarm value and 26-bit periodic alarm value
•automatic reload of counter value
•counters can be stalled if the CPU is stalled or in OCD mode
3.8.3Watchdog Timers
The ESP32-C3 family contains three watchdog timers: one in each of the two timer groups (called Main System
Watchdog Timers, or MWDT) and one in the RTC module (called the RTC Watchdog Timer, or RWDT).
During the flash boot process, RWDT and the MWDT in timer group 0 (TIMG0) are enabled automatically in order
to detect and recover from booting errors.
Watchdog timers have the following features:
•four stages, each with a programmable timeout value. Each stage can be configured, enabled and
disabled separately
•interrupt, CPU reset, or core reset for MWDT upon expiry of each stage; interrupt, CPU reset, core reset, or
system reset for RWDT upon expiry of each stage
•32-bit expiry counter
•write protection, to prevent RWDT and MWDT configuration from being altered inadvertently
•flash boot protection
If the boot process from an SPI flash does not complete within a predetermined period of time, the
watchdog will reboot the entire main system.
3.9Cryptographic Hardware Accelerators
ESP32-C3 family is equipped with hardware accelerators of general algorithms, such as AES-128/AES-256 (FIPS
PUB 197), ECB/CBC/OFB/CFB/CTR (NIST SP 800-38A), SHA1/SHA224/SHA256 (FIPS PUB 180-4), RSA3072,
and ECC. The chip also supports independent arithmetic, such as Big Integer Multiplication and Big Integer
Modular Multiplication. The maximum operation length for RSA and Big Integer Modular Multiplication is 3072
bits. The maximum factor length for Big Integer Multiplication is 1536 bits.
Espressif Systems24
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
3.10Physical Security Features
•transparent external flash encryption (AES-XTS algorithm) with software inaccessible key prevents
unauthorized readout of user application code or data.
•secure boot feature uses a hardware root of trust to ensure only signed firmware (with RSA-PSS signature)
can be booted.
•HMAC module can use a software inaccessible MAC key to generate MAC signatures for identity
verification and other purposes.
•Digital Signature module can use a software inaccessible secure key to generate RSA signatures for identity
verification.
•World Controller provides two running environments for software. All hardware and software resources are
sorted to two groups, and placed in either secure or general world. The secure world cannot be accessed
by hardware in the general world, thus establishing a security boundary.
3.11Peripheral Pin Configurations
Table 9: Peripheral Pin Configurations
InterfaceSignalPinFunction
ADCADC1_CH0XTAL_32K_PTwo 12-bit SAR ADCs
ADC1_CH1XTAL_32K_N
ADC1_CH2GPIO2
ADC1_CH3GPIO3
ADC1_CH4MTMS
ADC2_CH0MTDI
JTAGMTDIMTDIJTAG for software debugging
MTCKMTCK
MTMSMTMS
MTDOMTDO
UARTU0RXD_inAny GPIO pinsTwo UART channels with hardware flow control
and GDMAU0CTS_in
U0DSR_in
U0TXD_out
U0RTS_out
U0DTR_out
U1RXD_in
U1CTS_in
U1DSR_in
U1TXD_out
U1RTS_out
U1DTR_out
Espressif Systems25
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

3  Functional Description
InterfaceSignalPinFunction
I2CI2CEXT0_SCL_inAny GPIO pinsOne I2C channel in slave or master mode
I2CEXT0_SDA_in
I2CEXT1_SCL_in
I2CEXT1_SDA_in
I2CEXT0_SCL_out
I2CEXT0_SDA_out
I2CEXT1_SCL_out
I2CEXT1_SDA_out
LED PWMledc_ls_sig_out0~5Any GPIO pinsSix independent PWM channels
I2SI2S0O_BCK_inAny GPIO pinsStereo input and output from/to the audiocodec
I2S_MCLK_in
I2SO_WS_in
I2SI_SD_in
I2SI_BCK_in
I2SI_WS_in
I2SO_BCK_out
I2S_MCLK_out
I2SO_WS_out
I2SO_SD_out
I2SI_BCK_out
I2SI_WS_out
I2SO_SD1_out
Remote Control
Peripheral
RMT_SIG_IN0~1Any GPIO pinsTwo channels for an IR transceiver of various
waveformsRMT_SIG_OUT0~1
SPI0/1SPICLK_out_muxSPICLKSupport Standard SPI, Dual SPI, Quad SPI, and
QPI that allow connection to external flashSPICS0_outSPICS0
SPICS1_outAny GPIO pins
SPID_in/_outSPID
SPIQ_in/_outSPIQ
SPIWP_in/_outSPIWP
SPIHD_in/_outSPIHD
SPI2FSPICLK_in/_out_muxAny GPIO pins
•Master mode and slave mode of SPI, Dual
SPI, Quad SPI, and QPI
•Connection to external flash, RAM, and
other SPI devices
•Four modes of SPI transfer format
•Configurable SPI frequency
•64-byte FIFO or GDMA buffer
FSPICS0_in/_out
FSPICS1~5_out
FSPID_in/_out
FSPIQ_in/_out
FSPIWP_in/_out
FSPIHD_in/_out
USB Serial/JTAGUSB_D+GPIO19USB-to-serial converter, and USB-to-JTAG
converter
USB_D-GPIO18
TWAItwai_rxAny GPIO pinsCompatible with ISO 11898-1 protocol
twai_tx
twai_bus_off_on
twai_clkout
Espressif Systems26
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

4  Electrical Characteristics
4.Electrical Characteristics
4.1Absolute Maximum Ratings
Stresses beyond the absolute maximum ratings listed in the table below may cause permanent damage to the
device. These are stress ratings only, and do not refer to the functional operation of the device.
Table 10: Absolute Maximum Ratings
SymbolParameterMinMaxUnit
VDDA,  VDD3P3,  VDD3P3_RTC,
VDD3P3_CPU, VDD_SPI
Voltage applied to power supply pins
per power domain
–0.33.6V
T
ST ORE
Storage temperature–40150°C
4.2Recommended Operating Conditions
Table 11: Recommended Operating Conditions
SymbolParameterMinTypMaxUnit
VDDA, VDD3P3Voltage applied to power supply
3.03.33.6V
VDD3P3_RTCpins per power domain
VDD_SPI (working as
input power supply)
1
—3.03.33.6V
VDD3P3_CPU
2
Voltage applied to power supply pin3.03.33.6V
I
V DD
3
Current delivered by external power supply0.5——A
T
A
Ambient
temperature
ESP32-C3
–40—
105
°CESP32-C3FN485
ESP32-C3FH4105
1
For more information, please refer to Section2.3Power Scheme.
2
When VDD_SPI is used to drive peripherals, VDD3P3_CPU should comply with the peripherals’ specifica-
tions. For more information, please refer to Table
12.
3
If you use a single power supply, the recommended output current is 500 mA or more.
4.3VDD_SPI Output Characteristics
Table 12: VDD_SPI Output Characteristics
SymbolParameterTypUnit
R
SP I
On-resistance in 3.3 V mode7.5Ω
In real-life applications, when VDD_SPI works in 3.3 V output mode, VDD3P3_CPU may be affected
by R
SP I
. For example, when VDD3P3_CPU is used to drive a 3.3 V flash, it should comply with the
following specifications:
VDD3P3_CPU &gt; VDD_flash_min + I_flash_max*R
SP I
Among which, VDD_flash_min is the minimum operating voltage of the flash, and I_flash_max the
maximum current.
For more information, please refer to section
2.3Power Scheme.
Espressif Systems27
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

4  Electrical Characteristics
4.4DC Characteristics (3.3 V, 25 °C)
Table 13: DC Characteristics (3.3 V, 25 °C)
SymbolParameterMinTypMaxUnit
C
IN
Pin capacitance—2—pF
V
IH
High-level input voltage0.75 × VDD
1
—VDD
1
+ 0.3V
V
IL
Low-level input voltage–0.3—0.25 × VDD
1
V
I
IH
High-level input current——50nA
I
IL
Low-level input current——50nA
V
OH
2
High-level output voltage0.8 × VDD
1
——V
V
OL
2
Low-level output voltage——0.1 × VDD
1
V
I
OH
High-level source current (VDD
1
= 3.3 V,
V
OH
&gt;= 2.64 V, PAD_DRIVER = 3)
—40—mA
I
OL
Low-level sink current (VDD
1
= 3.3 V, V
OL
=
0.495 V, PAD_DRIVER = 3)
—28—mA
R
P U
Pull-up resistor—45—kΩ
R
P D
Pull-down resistor—45—kΩ
V
IH_nRST
Chip reset release voltage0.75 × VDD
1
—VDD
1
+ 0.3V
V
IL_nRST
Chip reset voltage–0.3—0.25 × VDD
1
V
1
VDD is the I/O voltage for a particular power domain of pins.
2
V
OH
and V
OL
are measured using high-impedance load.
4.5ADC Characteristics
Table 14: ADC Characteristics
SymbolParameterMinMaxUnit
DNL (Differential nonlinearity)
1
ADC connected to an external
–77LSB
100 nF capacitor; DC signal input;
INL (Integral nonlinearity)
ambient temperature at 25 °C;
–1212LSB
Wi-Fi off
Sampling rate——100Ksps
Effective Range
ATTEN00750mV
ATTEN101050mV
ATTEN201300mV
ATTEN302500mV
1
To get better DNL results, you can sample multiple times and apply a filter, or calculate the average value.
4.6Current Consumption
The current consumption measurements are taken with a 3.3 V supply at 25 °C of ambient temperature at the RF
port. All transmitters’ measurements are based on a 100% duty cycle.
Espressif Systems28
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

4  Electrical Characteristics
Table 15: Current Consumption Depending on RF Modes
Work modeDescription
Peak
(mA)
Active (RF working)
TX
802.11b, 1 Mbps, @21 dBm335
802.11g, 54 Mbps, @19 dBm285
802.11n, HT20, MCS 7, @18.5 dBm276
802.11n, HT40, MCS 7, @18.5 dBm278
RX
802.11b/g/n, HT2084
802.11n, HT4087
Table 16: Current Consumption Depending on Work Modes
Work modeDescriptionTypUnit
Modem-sleep
1, 2
The CPU is
powered on
3
160 MHz20mA
80 MHz15mA
Light-sleep—130μA
Deep-sleepRTC timer + RTC memory5μA
Power offCHIP_PU is set to low level, the chip is powered off1μA
1
The current consumption figures in Modem-sleep mode are for cases where the CPU is powered on and
the cache idle.
2
When Wi-Fi is enabled, the chip switches between Active and Modem-sleep modes. Therefore, current
consumption changes accordingly.
3
In Modem-sleep mode, the CPU frequency changes automatically. The frequency depends on the CPU
load and the peripherals used.
4.7Reliability
Table 17: Reliability Qualifications
Test ItemTest ConditionsTest Standard
HTOL (High Temperature
Operating Life)
125 °C, 1000 hoursJESD22-A108
ESD (Electro-Static
Discharge Sensitivity)
HBM (Human Body Mode)
1
± 2000 VJESD22-A114
CDM (Charge Device Mode)
2
± 1000 VJESD22-C101F
Latch up
Current trigger ± 200 mA
JESD78
Voltage trigger 1.5 × VDD
max
Preconditioning
Bake 24 hours @125 °C
Moisture soak (level 3: 192 hours @30 °C, 60% RH)
IR reflow solder: 260 + 0 °C, 20 seconds, three times
J-STD-020, JESD47,
JESD22-A113
TCT (Temperature Cycling
Test)
–65 °C / 150 °C, 500 cyclesJESD22-A104
uHAST (Highly
Accelerated Stress Test,
unbiased)
130 °C, 85% RH, 96 hoursJESD22-A118
Cont’d on next page
Espressif Systems29
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

4  Electrical Characteristics
Table 17 – cont’d from previous page
Test ItemTest ConditionsTest Standard
HTSL (High Temperature
Storage Life)
150 °C, 1000 hoursJESD22-A103
LTSL (Low Temperature
Storage Life)
– 40 °C, 1000 hoursJESD22-A119
1
JEDEC document JEP155 states that 500 V HBM allows safe manufacturing with a standard ESD control process.
2
JEDEC document JEP157 states that 250 V CDM allows safe manufacturing with a standard ESD control process.
4.8Wi­Fi Radio
Table 18: Wi­Fi Frequency
MinTypMax
Parameter
(MHz)(MHz)(MHz)
Center frequency of operating channel2412—2484
4.8.1Wi­Fi RF Transmitter (TX) Specifications
Table 19: TX Power with Spectral Mask and EVM Meeting 802.11 Standards
MinTypMax
Rate
(dBm)(dBm)(dBm)
802.11b, 1 Mbps—21.0—
802.11b, 11 Mbps—21.0—
802.11g, 6 Mbps—21.0—
802.11g, 54 Mbps—19.0—
802.11n, HT20, MCS0—20.0—
802.11n, HT20, MCS7—18.5—
802.11n, HT40, MCS0—20.0—
802.11n, HT40, MCS7—18.5—
Table 20: TX EVM Test
MinTypSL
1
Rate
(dB)(dB)(dB)
802.11b, 1 Mbps, @21 dBm—–24.5–10
802.11b, 11 Mbps, @21 dBm—–25.0–10
802.11g, 6 Mbps, @21 dBm—–23.0–5
802.11g, 54 Mbps, @19 dBm—–27.5–25
802.11n, HT20, MSC0, @20 dBm—–22.5–5
802.11n, HT20, MSC7, @18.5 dBm—–29.0–27
802.11n, HT40, MSC0, @20 dBm—–22.5–5
802.11n, HT40, MSC7, @18.5 dBm—–28.0–27
1
SL stands for standard limit value.
Espressif Systems30
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

4  Electrical Characteristics
4.8.2Wi­Fi RF Receiver (RX) Specifications
Table 21: RX Sensitivity
MinTypMax
Rate
(dBm)(dBm)(dBm)
802.11b, 1 Mbps—–98.4—
802.11b, 2 Mbps—–96.0—
802.11b, 5.5 Mbps—–93.0—
802.11b, 11 Mbps—–88.6—
802.11g, 6 Mbps—–93.8—
802.11g, 9 Mbps—–92.2—
802.11g, 12 Mbps—–91.0—
802.11g, 18 Mbps—–88.4—
802.11g, 24 Mbps—–85.8—
802.11g, 36 Mbps—–82.0—
802.11g, 48 Mbps—–78.0—
802.11g, 54 Mbps—–76.6—
802.11n, HT20, MCS0—–93.6—
802.11n, HT20, MCS1—–90.8—
802.11n, HT20, MCS2—–88.4—
802.11n, HT20, MCS3—–85.0—
802.11n, HT20, MCS4—–81.8—
802.11n, HT20, MCS5—–77.8—
802.11n, HT20, MCS6—–76.0—
802.11n, HT20, MCS7—–74.8—
802.11n, HT40, MCS0—–90.0—
802.11n, HT40, MCS1—–88.0—
802.11n, HT40, MCS2—–85.2—
802.11n, HT40, MCS3—–82.0—
802.11n, HT40, MCS4—–78.8—
802.11n, HT40, MCS5—–74.6—
802.11n, HT40, MCS6—–73.0—
802.11n, HT40, MCS7—–71.4—
Table 22: Maximum RX Level
MinTypMax
Rate
(dBm)(dBm)(dBm)
802.11b, 1 Mbps—5—
802.11b, 11 Mbps—5—
802.11g, 6 Mbps—5—
802.11g, 54 Mbps—0—
802.11n, HT20, MCS0—5—
802.11n, HT20, MCS7—0—
Cont’d on next page
Espressif Systems31
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

4  Electrical Characteristics
Table 22 – cont’d from previous page
MinTypMax
Rate
(dBm)(dBm)(dBm)
802.11n, HT40, MCS0—5—
802.11n, HT40, MCS7—0—
Table 23: RX Adjacent Channel Rejection
MinTypMax
Rate
(dB)(dB)(dB)
802.11b, 1 Mbps—35—
802.11b, 11 Mbps—35—
802.11g, 6 Mbps—31—
802.11g, 54 Mbps—20—
802.11n, HT20, MSC0—31—
802.11n, HT20, MSC7—16—
802.11n, HT40, MSC0—25—
802.11n, HT40, MSC7—11—
4.9Bluetooth LE Radio
Table 24: Bluetooth LE Frequency
MinTypMax
Parameter
(MHz)(MHz)(MHz)
Center frequency of operating channel2402—2480
4.9.1Bluetooth LE RF Transmitter (TX) Specifications
Table 25: Transmitter Characteristics ­ Bluetooth LE 1 Mbps
ParameterDescriptionMinTypMaxUnit
RF transmit power
RF power control range–27.00018.00dBm
Gain control step—3.00—dB
Carrier frequency offset and drift
Max|f
n
|
n=0,1,2, ..k
—17.00—kHz
Max|f
0−
f
n
|—1.75—kHz
Max
|
f
n−
f
n−5
|—1.46—kHz
|f
1−
f
0
|—0.80—kHz
Modulation characteristics
∆f1
avg
—250.00—kHz
Min∆f2
max
(for at least
99.9% of all∆f2
max
)
—190.00—kHz
∆f2
avg
/∆f1
avg
—0.83——
In-band spurious emissions
± 2 MHz offset—–37.62—dBm
± 3 MHz offset—–41.95—dBm
± &gt; 3 MHz offset—–44.48—dBm
Espressif Systems32
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

4  Electrical Characteristics
Table 26: Transmitter Characteristics ­ Bluetooth LE 2 Mbps
ParameterDescriptionMinTypMaxUnit
RF transmit power
RF power control range–27.00018.00dBm
Gain control step—3.00—dB
Carrier frequency offset and drift
Max|f
n
|
n=0,1,2, ..k
—20.80—kHz
Max|f
0−
f
n
|—1.30—kHz
Max|f
n−
f
n−5
|—1.33—kHz
|f
1−
f
0
|—0.70—kHz
Modulation characteristics
∆f1
avg
—498.00—kHz
Min∆f2
max
(for at least
99.9% of all∆f2
max
)
—430.00—kHz
∆f2
avg
/∆f1
avg
—0.93——
In-band spurious emissions
± 4 MHz offset—–43.55—dBm
± 5 MHz offset—–45.26—dBm
± &gt; 5 MHz offset—–45.26—dBm
Table 27: Transmitter Characteristics ­ Bluetooth LE 125 Kbps
ParameterDescriptionMinTypMaxUnit
RF transmit power
RF power control range–27.00018.00dBm
Gain control step—3.00—dB
Carrier frequency offset and drift
Max|f
n
|
n=0,1,2, ..k
—17.50—kHz
Max|f
0−
f
n
|—0.45—kHz
|f
n−
f
n−3
|—0.70—kHz
|f
0−
f
3
|—0.30—kHz
Modulation characteristics
∆f1
avg
—250.00—kHz
Min
∆
f
1
max
(for at least
99.9% of all∆f2
max
)
—235.00—kHz
In-band spurious emissions
± 2 MHz offset—–37.90—dBm
± 3 MHz offset—–41.00—dBm
± &gt; 3 MHz offset—–42.50—dBm
Table 28: Transmitter Characteristics ­ Bluetooth LE 500 Kbps
ParameterDescriptionMinTypMaxUnit
RF transmit power
RF power control range–27.00018.00dBm
Gain control step—3.00—dB
Carrier frequency offset and drift
Max|f
n
|
n=0,1,2, ..k
—17.00—kHz
Max|f
0−
f
n
|—0.88—kHz
|f
n−
f
n−3
|—1.00—kHz
|f
0−
f
3
|—0.20—kHz
Modulation characteristics
∆f2
avg
—208.00—kHz
Min∆f2
max
(for at least
99.9% of all∆f2
max
)
—190.00—kHz
Cont’d on next page
Espressif Systems33
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

4  Electrical Characteristics
Table 28 – cont’d from previous page
ParameterDescriptionMinTypMaxUnit
In-band spurious emissions
± 2 MHz offset—–37.90—dBm
± 3 MHz offset—–41.30—dBm
± &gt; 3 MHz offset—–42.80—dBm
4.9.2Bluetooth LE RF Receiver (RX) Specifications
Table 29: Receiver Characteristics ­ Bluetooth LE 1 Mbps
ParameterDescriptionMinTypMaxUnit
Sensitivity @30.8% PER——–97—dBm
Maximum received signal @30.8% PER——5—dBm
Co-channel C/I——8—dB
Adjacent channel selectivity C/I
F = F0 + 1 MHz—–3—dB
F = F0 – 1 MHz—–4—dB
F = F0 + 2 MHz—–29—dB
F = F0 – 2 MHz—–31—dB
F = F0 + 3 MHz—–33—dB
F = F0 – 3 MHz—–27—dB
F≥F0 + 4 MHz—–29—dB
F≤F0 – 4 MHz—–38—dB
Image frequency——–29—dB
Adjacent channel to image frequency
F = F
image
+ 1 MHz—–41—dB
F = F
image
– 1 MHz—–33—dB
Out-of-band blocking performance
30 MHz~2000 MHz—–5—dBm
2003 MHz~2399 MHz—–18—dBm
2484 MHz~2997 MHz—–15—dBm
3000 MHz~12.75 GHz—–5—dBm
Intermodulation——–30—dBm
Table 30: Receiver Characteristics ­ Bluetooth LE 2 Mbps
ParameterDescriptionMinTypMaxUnit
Sensitivity @30.8% PER——–93—dBm
Maximum received signal @30.8% PER——5—dBm
Co-channel C/I——10—dB
Adjacent channel selectivity C/I
F = F0 + 2 MHz—–7—dB
F = F0 – 2 MHz—–7—dB
F = F0 + 4 MHz—–28—dB
F = F0 – 4 MHz—–26—dB
F = F0 + 6 MHz—–26—dB
F = F0 – 6 MHz—–27—dB
F≥F0 + 8 MHz—–29—dB
F≤F0 – 8 MHz—–28—dB
Cont’d on next page
Espressif Systems34
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

4  Electrical Characteristics
Table 30 – cont’d from previous page
ParameterDescriptionMinTypMaxUnit
Image frequency——–28—dB
Adjacent channel to image frequency
F = F
image
+ 2 MHz—–26—dB
F = F
image
– 2 MHz—–7—dB
Out-of-band blocking performance
30 MHz~2000 MHz—–5—dBm
2003 MHz~2399 MHz—–19—dBm
2484 MHz~2997 MHz—–16—dBm
3000 MHz~12.75 GHz—–5—dBm
Intermodulation——–29—dBm
Table 31: Receiver Characteristics ­ Bluetooth LE 125 Kbps
ParameterDescriptionMinTypMaxUnit
Sensitivity @30.8% PER——–105—dBm
Maximum received signal @30.8% PER——5—dBm
Co-channel C/I——3—dB
Adjacent channel selectivity C/I
F = F0 + 1 MHz—–6—dB
F = F0 – 1 MHz—–6—dB
F = F0 + 2 MHz—–33—dB
F = F0 – 2 MHz—–43—dB
F = F0 + 3 MHz—–37—dB
F = F0 – 3 MHz—–47—dB
F≥F0 + 4 MHz—–40—dB
F≤F0 – 4 MHz—–50—dB
Image frequency——–40—dB
Adjacent channel to image frequency
F = F
image
+ 1 MHz—–50—dB
F = F
image
– 1 MHz—–37—dB
Table 32: Receiver Characteristics ­ Bluetooth LE 500 Kbps
ParameterDescriptionMinTypMaxUnit
Sensitivity @30.8% PER——–100—dBm
Maximum received signal @30.8% PER——5—dBm
Co-channel C/I——3—dB
Adjacent channel selectivity C/I
F = F0 + 1 MHz—–2—dB
F = F0 – 1 MHz—–3—dB
F = F0 + 2 MHz—–32—dB
F = F0 – 2 MHz—–33—dB
F = F0 + 3 MHz—–23—dB
F = F0 – 3 MHz—–40—dB
F≥F0 + 4 MHz—–34—dB
F≤F0 – 4 MHz—–44—dB
Image frequency——–34—dB
Cont’d on next page
Espressif Systems35
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

4  Electrical Characteristics
Table 32 – cont’d from previous page
ParameterDescriptionMinTypMaxUnit
Adjacent channel to image frequency
F = F
image
+ 1 MHz—–46—dB
F = F
image
– 1 MHz—–23—dB
Espressif Systems36
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

5  Package Information
5.Package Information
Figure 8: QFN32 (5×5 mm) Package
Note:
•For the source file ofrecommendedPCBlandpattern(dxf), you can view it withAutodeskViewer;
•For information about tape, reel, and product marking, please refer toEspressif Chip-Packing Information.
Espressif Systems37
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

Revision History
Revision History
DateVersionRelease Notes
2021-05-28V1.0
•Updated power modes;
•Updated Section2.4Strapping Pins;
•Updated some clock names and their frequencies in Section3.2System
Clocks;
•Added clarification about ADC1 and ADC2 in Section3.3.1Analog-to-
Digital Converter (ADC);
•Updated the default configuration of U0RXD andU0TXD after reset in
TableGeneral Purpose Input / Output Interface (GPIO);
•Updated sampling rate in TableADC Characteristics;
•Updated TableReliability;
•Added the link to recommended PCB land pattern in Chapter5Package
Information.
2021-04-23V0.8UpdatedWi-Fi RadioandBluetooth LE Radiodata.
2021-04-07V0.7
•Updated information aboutUSB Serial/JTAG Controller;
•Added GPIO2 to Section2.4Strapping Pins;
•Updated FigureAddress Mapping Structure;
•Added TableGeneral Purpose Input / Output Interface (GPIO)and Table
General Purpose Input / Output Interface (GPIO)in Section3.4.1General
Purpose Input / Output Interface (GPIO);
•Updated information about SPI2 in Section3.4.2Serial Peripheral Inter-
face (SPI);
•Updated fixed-priority channel scheme in Section3.4.8General DMA
Controller
;
•Updated TableReliability.
2021-01-18V0.6
•Clarified that of the 400 KB SRAM, 16 KB is configured as cache;
•Updated maximum value to standard limit value in TableWi-Fi RF Trans-
mitter (TX) Specificationsin Section4.8.1Wi-Fi RF Transmitter (TX) Spec-
ifications
.
Espressif Systems38
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

Revision History
DateVersionRelease Notes
2021-01-13V0.5
•Updated information about Wi-Fi;
•Added connection between embedded flash ports and chip pins to ta-
ble notes in Section2.2Pin Description;
•Updated FigureESP32-C3 Family Power Scheme, added FigureESP32-
C3 Family Power-up and Reset Timingand TablePower Schemein Sec-
tion2.3Power Scheme;
•Added FigureSetup and Hold Times for the Strapping Pinand Table
Strapping Pinsin Section2.4Strapping Pins;
•Updated TablePeripheral Pin Configurationsin Section3.11Peripheral
Pin Configurations;
•Added Chapter4Electrical Characteristics;
•Added Chapter5Package Information.
2020-11-27V0.4Preliminary version.
Espressif Systems39
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

Solutions, Documentation and Legal Information
Solutions, Documentation and Legal Information
Must­Read Documents
•ESP32-C3 Technical Reference Manual
•ESP32-C3 Hardware Design Guidelines
•ESP-IDF Programming Guide
•Espressif Product Ordering Information
•Certificates
•NotificationSubscription
Sales and Technical Support
•SalesQuestions
•TechnicalInquiries
•GetSamples
Developer Zone
•ESP32Forum
•GitHub
•Courses
•Videos
Products
•SoCs
•Modules
•DevKits
Must­Have Resources
•SDKsandDemos
•APPs
•Tools
•AT
Espressif Systems40
Submit Documentation Feedback
ESP32-C3 Family Datasheet V1.0

www.espressif.com
Disclaimer and Copyright Notice
Information in this document, including URL references, is subject to change without notice.
ALL THIRD PARTY’S INFORMATION IN THIS DOCUMENT IS PROVIDED AS IS WITH NO
WARRANTIES TO ITS AUTHENTICITY AND ACCURACY.
NO WARRANTY IS PROVIDED TO THIS DOCUMENT FOR ITS MERCHANTABILITY, NON-
INFRINGEMENT, FITNESS FOR ANY PARTICULAR PURPOSE, NOR DOES ANY WARRANTY
OTHERWISE ARISING OUT OF ANY PROPOSAL, SPECIFICATION OR SAMPLE.
All liability, including liability for infringement of any proprietary rights, relating to use of information
in this document is disclaimed. No licenses express or implied, by estoppel or otherwise, to any
intellectual property rights are granted herein.
The Wi-Fi Alliance Member logo is a trademark of the Wi-Fi Alliance. The Bluetooth logo is a
registered trademark of Bluetooth SIG.
All trade names, trademarks and registered trademarks mentioned in this document are property
of their respective owners, and are hereby acknowledged.
Copyright © 2021 Espressif Systems (Shanghai) Co., Ltd. All rights reserved.