

ESP32-C3 Series
DatasheetVersion 2.4
Ultra-Low-Power SoC with RISC-V Single-Core CPU
2.4 GHz Wi-Fi (802.11b/g/n) and Bluetooth
®
5 (LE)
Optional flash in the chip’s package up to 8 MB
QFN32 (5×5 mm) package
Including:
ESP32-C3
ESP32-C3FN4 –Endoflife(EOL)
ESP32-C3FH4
ESP32-C3FH4AZ –
NotRecommendedforNewDesigns(NRND)
ESP32-C3FH4X – Recommended
ESP32-C3FH8X
www.espressif.com

Product Overview
ESP32-C3 is a low-power and highly-integrated MCU-based solution that supports 2.4 GHz Wi-Fi and
Bluetooth
®
Low Energy (Bluetooth LE).
The functional block diagram of the SoC is shown below.
Espressif ESP32-C3 Wi-Fi + Bluetooth
®
 Low Energy SoC
Power consumption
Normal
Low power consumption components capable of working in Deep-sleep mode
Wireless Digital Circuits
Wi-Fi MAC 
Wi-Fi 
Baseband
Bluetooth LE Link Controller
Bluetooth LE Baseband
CPU and Memory
RISC-V 32-bit
Microprocessor
JTAG
Cache
ROM
SRAM
Security
Flash 
Encryption
RSARNG
Digital 
Signature
SHAAES
HMAC
Secure Boot
RTC
RTC 
Memory
PMU
Peripherals
USB Serial/
JTAG
GPIO
UART
TWAI
®
General-
purpose 
Timers
I2S
I2C
LED PWM
SPI0/1
RMT
SPI2DIG ADC
System 
Timer
Temperature 
Sensor
RTC 
Watchdog 
Timer
GDMA
Main System 
Watchdog 
Timers
eFuse 
Controller
RF
2.4 GHz Balun + 
Switch
2.4 GHz 
Receiver
2.4 GHz 
Transmitter
RF 
Synthesizer
Fast RC 
Oscillator
External 
Main Clock
Phase Lock 
Loop
Super 
Watchdog
World 
Controller
Debug 
Assistant
Brownout Detector
ESP32-C3 Functional Block Diagram
For more information on power consumption, see Section4.1.3.6Power Management Unit.
The ESP32-C3chip seriesis a member of the ESP32-C3chip series group. For more information about this
chip series group, see
ESP32-C3 Chip Series Group Overview.
Espressif Systems2
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Features
Wi-Fi
•Complies with IEEE 802.11b/g/n
•Supports 20 MHz and 40 MHz bandwidth in 2.4 GHz band
•1T1R mode with data rate up to 150 Mbps
•Wi-Fi Multimedia (WMM)
•TX/RX A-MPDU, TX/RX A-MSDU
•Immediate Block ACK
•Fragmentation and defragmentation
•Transmit opportunity (TXOP)
•Automatic Beacon monitoring (hardware TSF)
•4 × virtual Wi-Fi interfaces
•Simultaneous support for Infrastructure BSS in Station mode, SoftAP mode, Station + SoftAP mode, and
promiscuous mode
Note that when ESP32-C3 scans in Station mode, the SoftAP channel will change along with the Station
channel
•Antenna diversity
•802.11mc FTM
Bluetooth
®
•Bluetooth LE: Bluetooth 5, Bluetooth mesh
•High power mode with up to 20 dBm transmission power
•Speed: 125 Kbps, 500 Kbps, 1 Mbps, 2 Mbps
•Advertising extensions
•Multiple advertisement sets
•Channel selection algorithm #2
•Internal co-existence mechanism between Wi-Fi and Bluetooth to share the same antenna
CPU and Memory
•32-bit RISC-V single-core processor
•Clock speed: up to 160 MHz
•CoreMark
®
score:
–1 core at 160 MHz: 483.27 CoreMark; 3.02 CoreMark/MHz
•General DMA controller (GDMA), with 3 transmit channels and 3 receive channels
Espressif Systems3
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

•ROM: 384 KB
•SRAM: 400 KB (16 KB for cache)
•SRAM in RTC: 8 KB
•4096-bit eFuse memory, up to 1792 bits for users
•In-package flash(see details in Chapter1ESP32-C3 Series Comparison)
•Supported SPI protocols: SPI, Dual SPI, Quad SPI, and QPI interfaces that allow connection to multiple
off-package flashand other SPI devices
•Access to flash accelerated by cache
•Supports flash in-Circuit Programming (ICP)
Peripherals
•Programmable GPIOs
–22 for SP32-C3, ESP32-C3FH4, ESP32-C3FN4 and ESP32-C3FH8X
*3 strapping GPIOs
*6 GPIOs allocated for in-package flash
–16 for ESP32-C3FH4X and ESP32-C3FH4AZ
*3 strapping GPIOs
•Connectivity interfaces:
–Two UARTs
–Three SPI
–I2C
–I2S
–Full-speed USB Serial/JTAG controller
–TWAI
®
controller compatible with ISO 11898-1 (CAN Specification 2.0)
–LED PWM controller, with up to 6 channels
–Remote control peripheral, with 2 transmit channels and 2 receive channels
•Analog signal processing:
–Two 12-bit SAR ADCs, up to 6 channels
–Temperature sensor
•Timers:
–Two 54-bit general-purpose timers
–Three digital watchdog timers
–Analog watchdog timer
–52-bit system timer
Espressif Systems4
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

–XTAL32K wachtog timer
Power Management
•Fine-resolution power control, including clock frequency, duty cycle, Wi-Fi operating modes, and
individual internal component control
•Four power modes designed for typical scenarios: Active, Modem-sleep, Light-sleep, Deep-sleep
•Power consumption in Deep-sleep mode is 5μA
•RTC memory remains powered on in Deep-sleep mode
Security
•Secure boot - permission control on accessing internal and external memory
•Flash encryption - memory encryption and decryption
•Cryptographic hardware acceleration:
–AES-128/256 (FIPS PUB 197)
–SHA Accelerator (FIPS PUB 180-4)
–RSA Accelerator
–Random Number Generator (RNG)
–HMAC
–Digital signature
•Clock glitch detection
RF Module
•Antenna switches, RF balun, power amplifier, low-noise receive amplifier
•Up to +21 dBm of power for an 802.11b transmission
•Up to +20 dBm of power for an 802.11n transmission
•Up to -105 dBm of sensitivity for Bluetooth LE receiver (125 Kbps)
Applications
With low power consumption, ESP32-C3 is an ideal choice for IoT devices in the following areas:
•Smart Home
•Industrial Automation
•Health Care
•Consumer Electronics
•Smart Agriculture
•POS Machines
•Service Robot
•Audio Devices
•Generic Low-power IoT Sensor Hubs
•Generic Low-power IoT Data Loggers
Espressif Systems5
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Contents
Note:
Check the link or the QR code to make sure that you use the latest version of this document:
https://www.espressif.com/documentation/esp32-c3_datasheet_en.pdf
Contents
Product Overview2
Features3
Applications5
1  ESP32-C3 Series Comparison12
1.1    Nomenclature12
1.2   Comparison12
1.3   Chip Revision13
2  Pins14
2.1   Pin Layout14
2.2   Pin Overview16
2.3   IO Pins19
2.3.1   IO MUX Functions19
2.3.2  Analog Functions21
2.3.3  Restrictions for GPIOs22
2.3.4  Peripheral Pin Assignment23
2.4   Analog Pins25
2.5   Power Supply26
2.5.1   Power Pins26
2.5.2  Power Scheme26
2.5.3  Chip Power-up and Reset27
2.6   Pin Mapping Between Chip and Flash29
3  Boot Configurations30
3.1   Chip Boot Mode Control31
3.2   ROM Messages Printing Control31
4  Functional Description33
4.1   System33
4.1.1   Microprocessor and Master33
4.1.1.1High-Performance CPU33
4.1.1.2    GDMA Controller33
4.1.2   Memory Organization33
4.1.2.1    Internal Memory34
4.1.2.2   External Memory35
Espressif Systems6
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Contents
4.1.2.3   Cache35
4.1.2.4   eFuse Controller35
4.1.3   System Components36
4.1.3.1    IO MUX and GPIO Matrix36
4.1.3.2   Reset36
4.1.3.3   Clock37
4.1.3.4   Interrupt Matrix37
4.1.3.5   System Timer38
4.1.3.6   Power Management Unit38
4.1.3.7   Timer Group40
4.1.3.8   Watchdog Timers40
4.1.3.9   XTAL32K Watchdog Timers (XTWDT)41
4.1.3.10   Permission Control41
4.1.3.11   World Controller (WCL)41
4.1.3.12   System Registers42
4.1.3.13   Debug Assistant42
4.1.4   Cryptography and Security Component43
4.1.4.1    AES Accelerator43
4.1.4.2   HMAC Accelerator43
4.1.4.3   RSA Accelerator44
4.1.4.4   SHA Accelerator44
4.1.4.5   Digital Signature44
4.1.4.6   External Memory Encryption and Decryption45
4.1.4.7   Random Number Generator45
4.1.4.8   Clock Glitch Detection45
4.2   Peripherals46
4.2.1   Connectivity Interface46
4.2.1.1    UART Controller46
4.2.1.2   SPI Controller46
4.2.1.3   I2C Controller47
4.2.1.4   I2S Controller47
4.2.1.5   USB Serial/JTAG Controller48
4.2.1.6   Two-wire Automotive Interface48
4.2.1.7   LED PWM Controller48
4.2.1.8   Remote Control Peripheral49
4.2.2  Analog Signal Processing49
4.2.2.1   SAR ADC49
4.2.2.2  Temperature Sensor50
4.3   Wireless Communication51
4.3.1   Radio51
4.3.1.1    2.4 GHz Receiver51
4.3.1.2   2.4 GHz Transmitter51
4.3.1.3   Clock Generator51
4.3.2  Wi-Fi52
4.3.2.1   Wi-Fi Radio and Baseband52
4.3.2.2  Wi-Fi MAC52
Espressif Systems7
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Contents
4.3.2.3   Networking Features53
4.3.3  Bluetooth LE53
4.3.3.1   Bluetooth LE PHY53
4.3.3.2   Bluetooth LE Link Controller53
5  Electrical Characteristics54
5.1   Absolute Maximum Ratings54
5.2   Recommended Operating Conditions54
5.3   VDD_SPI Output Characteristics55
5.4   DC Characteristics (3.3 V, 25 °C)55
5.5   ADC Characteristics56
5.6   Current Consumption56
5.6.1   RF Current Consumption in Active Mode56
5.6.2  Current Consumption in Other Modes57
5.7   Memory Specifications57
5.8   Reliability58
6  RF Characteristics59
6.1   Wi-Fi Radio59
6.1.1   Wi-Fi RF Transmitter (TX) Characteristics59
6.1.2   Wi-Fi RF Receiver (RX) Characteristics60
6.2   Bluetooth 5 (LE) Radio61
6.2.1   Bluetooth LE RF Transmitter (TX) Characteristics61
6.2.2  Bluetooth LE RF Receiver (RX) Characteristics63
7  Packaging66
ESP32-C3 Consolidated Pin Overview67
ESP32-C3 Chip Series Group Overview68
Datasheet Versioning69
Glossary70
Related Documentation and Resources71
Revision History72
Espressif Systems8
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

List of Tables
List of Tables
1-1  ESP32-C3 Series Comparison12
2-1  Pin Overview16
2-2 Power-Up Glitches on Pins18
2-3 Peripheral Signals Routed via IO MUX19
2-4 IO MUX Pin Functions19
2-5 Analog Signals Routed to Analog Functions21
2-6 Analog Functions21
2-7  Peripheral Pin Assignment24
2-8 Analog Pins25
2-9 Power Pins26
2-10 Voltage Regulators26
2-11 Description of Timing Parameters for Power-up and Reset28
2-12 Pin Mapping Between Chip and In-package Flash29
3-1  Default Configuration of Strapping Pins30
3-2 Description of Timing Parameters for the Strapping Pins30
3-3 Chip Boot Mode Control31
3-4 UART0 ROM Message Printing Control32
3-5 USB Serial/JTAG ROM Message Printing Control32
4-1  Components and Power Domains39
5-1  Absolute Maximum Ratings54
5-2 Recommended Operating Conditions54
5-3 VDD_SPI Internal and Output Characteristics55
5-4 DC Characteristics (3.3 V, 25 °C)55
5-5 ADC Characteristics56
5-6 ADC Calibration Results56
5-7  Wi-Fi Current Consumption Depending on RF Modes56
5-8 Current Consumption in Modem-sleep Mode57
5-9 Current Consumption in Low-Power Modes57
5-10 Flash Specifications57
5-11 Reliability Qualifications58
6-1  Wi-Fi Frequency59
6-2 TX Power with Spectral Mask and EVM Meeting 802.11 Standards59
6-3 TX EVM Test59
6-4 RX Sensitivity60
6-5 Maximum RX Level61
6-6 RX Adjacent Channel Rejection61
6-7  Bluetooth LE Frequency61
6-8 Transmitter Characteristics - Bluetooth LE 1 Mbps61
6-9 Transmitter Characteristics - Bluetooth LE 2 Mbps62
6-10 Transmitter Characteristics - Bluetooth LE 125 Kbps62
6-11 Transmitter Characteristics - Bluetooth LE 500 Kbps63
6-12 Receiver Characteristics - Bluetooth LE 1 Mbps63
6-13 Receiver Characteristics - Bluetooth LE 2 Mbps64
Espressif Systems9
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

List of Tables
6-14 Receiver Characteristics - Bluetooth LE 125 Kbps64
6-15 Receiver Characteristics - Bluetooth LE 500 Kbps65
Espressif Systems10
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

List of Figures
List of Figures
1-1  ESP32-C3 Series Nomenclature12
2-1  Pin Layout (Top View) for ESP32-C3, ESP32-C3FH4, ESP32-C3FN4, and ESP32-C3FH8X14
2-2 Pin Layout (Top View) for ESP32-C3FH4X and ESP32-C3FH4AZ15
2-3 ESP32-C3 Power Scheme27
2-4 Visualization of Timing Parameters for Power-up and Reset27
3-1  Visualization of Timing Parameters for the Strapping Pins31
4-1  Address Mapping Structure34
4-2 Components and Power Domains39
7-1  QFN32 (5×5 mm) Package66
Espressif Systems11
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

1  ESP32-C3 Series Comparison
1ESP32-C3 Series Comparison
1.1Nomenclature
ESP32-C3
ESP32-C3
F
F
H/N
H/N
*
*
Flash size (MB)
Flash temperature 
H: High temperature 
N: Normal temperature
In-package flash
Chip Series
AZ
AZ
X
X
Other identification code
Chip revision v1.1
Figure 1-1. ESP32-C3 Series Nomenclature
1.2Comparison
Table 1-1. ESP32-C3 Series Comparison
Part Number
1
In-Package Flash
3
Ambient Temp.
2
(°C)Package (mm)GPIO No.
6
Chip Revision
4
ESP32-C3
5
—40∼105QFN32 (5*5)22v0.4
ESP32-C3FN4
(EOL)
4 MB40∼85QFN32 (5*5)22v0.4
ESP32-C3FH44 MB40∼105QFN32 (5*5)22v0.4
ESP32-C3FH4AZ
(NRND)
4 MB40∼105QFN32 (5*5)16v0.4
ESP32-C3FH4X
(Recommended)
4 MB40∼105QFN32 (5*5)16v1.1
ESP32-C3FH8X
8 MB40∼105QFN32 (5*5)22v1.1
Espressif Systems12
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

1  ESP32-C3 Series Comparison
1
For details on chip marking and packing, see Section7Packaging.
2
Ambient temperature specifies the recommended temperature range of the environment immediately out-
side an Espressif chip.
3
For information about in-package flash, see also Section4.1.2.1Internal Memory. By default, the SPI flash
on the chip operates at a maximum clock frequency of 80 MHz and does not support the auto suspend
feature. If you have a requirement for a higher flash clock frequency of 120 MHz or if you need the flash
auto suspend feature, please contact us.
4
All chip revisions have the same SRAM size, but chip revision v1.1 has around 10 KB more available space
for users than chip revision v0.4. Chip revision v1.1 depends on specific ESP-IDF versions, as detailed in
CompatibilityAdvisoryforESP32-C3ChipRevisionv1.1. For how to identify chip revisions, please refer to
ESP32-C3 Series SoC Errata.
5
ESP32-C3 requires an SPI flash off the chip’s package. For details about SPI modes, see Section2.6Pin
Mapping Between Chip and Flash.
6
SPI0/SPI1 pins for flash connection are not bonded for variants with 16 GPIOs.
1.3Chip Revision
As shown in Table1-1ESP32-C3 Series Comparison, ESP32-C3 now has multiple chip revisions available on
the market.
For chip revision identification, ESP-IDF release that supports a specific chip revision, and errors fixed in each
chip revision, please refer to
ESP32-C3 Series SoC Errata.
Espressif Systems13
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
2Pins
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
111213
1415
16
2728293031322625
GPIO10
GPIO9GPIO8
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
VDDAVDDAXTAL_PXTAL_NU0TXDU0RXDGPIO19GPIO18
SPID
SPICLK
SPICS0
SPIWP
SPIHD
VDD_SPI
VDD3P3_CPU
ESP32-C3
33 GND
SPIQ
Figure 2-1. Pin Layout (Top View) for ESP32-C3, ESP32-C3FH4, ESP32-C3FN4, and ESP32-C3FH8X
Espressif Systems14
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
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
101112
13
1415
16
2728
293031322625
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
VDDAVDDAXTAL_PXTAL_NU0TXDU0RXDGPIO19GPIO18
NC
NC
NC
NC
NC
VDD_SPI
VDD3P3_CPU
ESP32-C3
33 GND
NC
Figure 2-2. Pin Layout (Top View) for ESP32-C3FH4X and ESP32-C3FH4AZ
Espressif Systems15
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
2.2Pin Overview
The ESP32-C3 chip integrates multiple peripherals that require communication with the outside world. To keep
the chip package size reasonably small, the number of available pins has to be limited. So the only way to
route all the incoming and outgoing signals is through pin multiplexing. Pin muxing is controlled via software
programmable registers (seeESP32-C3 Technical Reference Manual&gt; ChapterIO MUX and GPIO
Matrix).
All in all, the ESP32-C3 chip has the following types of pins:
•IO pinswith the following predefined sets of functions to choose from:
–EachIO pin has predefinedIO MUX functions– see Table2-4IO MUX Pin Functions
–SomeIO pins have predefinedanalog functions– see Table2-6Analog Functions
Predefined functionsmeans that each IO pin has a set of direct connections to certain signals from
on-chipperipherals. During run-time, the user can configure which peripheral signal from a predefined
set to connect to a certain pin at a certain time via memory mapped registers.
•Analog pinsthat have exclusively-dedicatedanalog functions– see Table2-8Analog Pins
•Power pinsthat supply power to the chip components and non-power pins – see Table2-9Power Pins
Table2-1Pin Overviewgives an overview of all the pins. For more information, see the respective sections for
each pin type below, orESP32-C3 Consolidated Pin Overview.
Table 2-1. Pin Overview
PinPinPinPin ProvidingPin Settings
5
Pin Function Sets
1
No.NameTypePower
2-4
At ResetAfter ResetIO MUXAnalog
1LNA_INAnalog
2VDD3P3Power
3VDD3P3Power
4XTAL_32K_PIOVDD3P3_RTCIO MUXAnalog
5XTAL_32K_NIOVDD3P3_RTCIO MUXAnalog
6GPIO2IOVDD3P3_RTCIEIEIO MUXAnalog
7CHIP_ENAnalog
8GPIO3IOVDD3P3_RTCIEIEIO MUXAnalog
9MTMSIOVDD3P3_RTCIEIO MUXAnalog
10MTDIIOVDD3P3_RTCIEIO MUXAnalog
11VDD3P3_RTCPower
12MTCKIOVDD3P3_CPUIE
6
IO MUX
13MTDOIOVDD3P3_CPUIEIO MUX
14GPIO8IOVDD3P3_CPUIEIEIO MUX
15GPIO9IOVDD3P3_CPUIE, WPUIE, WPUIO MUX
16GPIO10IOVDD3P3_CPUIEIO MUX
17VDD3P3_CPUPower
18VDD_SPI
8
PowerVDD3P3_CPUIO MUX
19SPIHDIOVDD_SPI / VDD3P3_CPUWPUIE, WPUIO MUX
20SPIWP
9
IOVDD_SPI / VDD3P3_CPUWPUIE, WPUIO MUX
Cont’d on next page
Espressif Systems16
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
Table 2-1 – cont’d from previous page
PinPinPinPin ProvidingPin Settings
5
Pin Function Sets
1
No.NameTypePower
2-4
At ResetAfter ResetIO MUXAnalog
21SPICS0IOVDD_SPI / VDD3P3_CPUWPUIE, WPUIO MUX
22SPICLKIOVDD_SPI / VDD3P3_CPUWPUIE, WPUIO MUX
23SPIDIOVDD_SPI / VDD3P3_CPUWPUIE, WPUIO MUX
24SPIQ
9
IOVDD_SPI / VDD3P3_CPUWPUIE, WPUIO MUX
25GPIO18IOVDD3P3_CPUIO MUXAnalog
26GPIO19IOVDD3P3_CPUUSB_PUIO MUXAnalog
27U0RXDIOVDD3P3_CPUIE, WPUIO MUX
28U0TXDIOVDD3P3_CPUWPU
7
IO MUX
29XTAL_NAnalog
30XTAL_PAnalog
31VDDAPower
32VDDAPower
33GNDPower
1.Boldmarks the pin function set in which a pin has its default function in the default boot mode. See Section3.1Chip Boot Mode
Control.
2.In columnPin Providing Power, regarding pins powered by VDD_SPI:
•Power actually comes from the internal power rail supplying power to VDD_SPI. For details, see Section2.5.2Power
Scheme.
3.In columnPin Providing Power, regarding pins powered by VDD3P3_CPU / VDD_SPI:
•Pin Providing Power (either VDD3P3_CPU or VDD_SPI) can be configured via a register, see
ESP32-C3 Technical Reference Manual&gt; ChapterIO MUX and GPIO Matrix.
4.The default drive strength for each pin is as follows:
•GPIO2, GPIO3, MTMS, and MTDI: 10 mA
•GPIO18, GPIO19: 40 mA
•All other pins: 20 mA
5.Column
Pin Settings
shows predefined settings at reset and after reset with the following abbreviations:
•IE – input enabled
•WPU – internal weak pull-up resistor enabled
•WPD – internal weak pull-down resistor enabled
•USB_PU – USB pull-up resistor enabled
–By default, the USB function is enabled for USB pins (i.e., GPIO18 and GPIO19), and the pin pull-up is decided by the
USB pull-up resistor. The USB pull-up resistor is controlled by USB_SERIAL_JTAG_DP/DM_PULLUP and the pull-up
value is controlled by USB_SERIAL_JTAG_PULLUP_VALUE. For details, see
ESP32-C3 Technical Reference Manual&gt;
ChapterUSB Serial/JTAG Controller).
–When the USB function is disabled, USB pins are used as regular GPIOs and the pin’s internal weak pull-up and
pull-down resistors are disabled by default (configurable by IO_MUX_FUN_
WPU/WPD). For details, see
ESP32-C3 Technical Reference Manual&gt; ChapterIO MUX and GPIO Matrix.
6.Depends on the value of EFUSE_DIS_PAD_JTAG
•0- default value. Input enabled, and internal weak pull-up resistor enabled (IE &amp; WPU)
•1- input enabled (IE)
7.Output enabled
8.By default VDD_SPI is the power supply pin for in-package and off-package flash. It can be reconfigured as a GPIO pin, if the chip
is connected to an off-package flash, and this flash is powered by an external power supply. For details about reconfiguration,
please refer to
ESP32-C3 Technical Reference Manual&gt; ChapterIO MUX and GPIO Matrix.
9.For ESP32-C3FH4AZ and ESP32-C3FH4X, pins within the frame (namely pin 19∼pin 24) are not bonded, and are labelled as ”not
connected (NC)”.
Espressif Systems17
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
Some pins have glitches during power-up. See details in Table2-2.
Table 2-2. Power-Up Glitches on Pins
PinGlitch
1
Typical Time Period(ns)
MTCKLow-level glitch5
MTDOLow-level glitch5
GPIO10Low-level glitch5
U0RXDLow-level glitch5
GPIO18High-level glitch50000
1
Low-level glitch: the pin is at a low level output status during the time
period;
High-level glitch: the pin is at a high level output status during the
time period;
Pull-down glitch: the pin is at an internal weak pulled-down status
during the time period;
Pull-up glitch: the pin is at an internal weak pulled-up status during
the time period.
Please refer to Table
5-4for detailed parameters about low/high-level
and pull-down/up.
Espressif Systems18
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
2.3IO Pins
2.3.1IO MUX Functions
The IO MUX allows multiple input/output signals to be connected to a single input/output pin. Each IO pin of
ESP32-C3 can be connected to one of the three signals (IO MUX functions, i.e., F0-F2), as listed in Table2-4
IO MUX Pin Functions.
Among the three sets of signals:
•Some are routed via the GPIO Matrix (GPIO0, GPIO1, etc.), which incorporates internal signal routing
circuitry for mapping signals programmatically. It gives the pin access to almost anyperipheralsignals.
However, the flexibility of programmatic mapping comes at a cost as it might affect the latency of routed
signals. For details about connecting to peripheral signals via GPIO Matrix, see
ESP32-C3 Technical Reference Manual&gt; ChapterIO MUX and GPIO Matrix.
•Some are directly routed from certain peripherals (U0TXD, MTCK, etc.), including UART0, JTAG, SPI0/1,
and SPI2 - see Table2-3Peripheral Signals Routed via IO MUX.
Table 2-3. Peripheral Signals Routed via IO MUX
Pin FunctionSignalDescription
U0TXDTransmit data
UART0 interface
U0RXDReceive data
MTCKTest clock
JTAG interface for debugging
MTDOTest Data Out
MTDITest Data In
MTMSTest Mode Select
SPIQMaster in, slave out
3.3 V SPI0/1 interface for connection to in-package or off-package flash
via the SPI bus. It supports 1-, 2-, 4-line SPI modes. See also Section
2.6Pin Mapping Between Chip and Flash
SPIDMaster out, slave in
SPIHDHold
SPIWPWrite protect
SPICLKClock
SPICS...Chip select
FSPIQMaster in, slave out
SPI2 interface for fast SPI connection. It supports 1-, 2-, 4-line SPI
modes
FSPIDMaster out, slave in
FSPIHDHold
FSPIWPWrite protect
FSPICLKClock
FSPICS0Chip select
Table2-4IO MUX Pin Functionsshows the IO MUX functions of IO pins.
Table 2-4. IO MUX Pin Functions
PinIO MUX /IO MUX Function
1, 2, 3
No.
GPIO
Name
2
F0Type
3
F1TypeF2Type
4GPIO0GPIO0I/O/TGPIO0I/O/T
Cont’d on next page
Espressif Systems19
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
Table 2-4 – cont’d from previous page
PinIO MUX /IO MUX Function
1, 2, 3
No.
GPIO
Name
2
F0Type
3
F1TypeF2Type
5GPIO1GPIO1I/O/TGPIO1I/O/T
6GPIO2GPIO2I/O/TGPIO2I/O/TFSPIQI1/O/T
8GPIO3GPIO3I/O/TGPIO3I/O/T
9GPIO4MTMSI1GPIO4I/O/TFSPIHDI1/O/T
10GPIO5MTDII1GPIO5I/O/TFSPIWPI1/O/T
12GPIO6MTCKI1GPIO6I/O/TFSPICLKI1/O/T
13GPIO7MTDOO/TGPIO7I/O/TFSPIDI1/O/T
14GPIO8GPIO8I/O/TGPIO8I/O/T
15GPIO9GPIO9I/O/TGPIO9I/O/T
16GPIO10GPIO10I/O/TGPIO10I/O/TFSPICS0I1/O/T
18GPIO11GPIO11I/O/TGPIO11I/O/T
19GPIO12SPIHDI1/O/TGPIO12I/O/T
20GPIO13SPIWPI1/O/TGPIO13I/O/T
21GPIO14SPICS0O/TGPIO14I/O/T
22GPIO15SPICLKO/TGPIO15I/O/T
23GPIO16SPIDI1/O/TGPIO16I/O/T
24GPIO17SPIQI1/O/TGPIO17I/O/T
25GPIO18GPIO18I/O/TGPIO18I/O/T
26GPIO19GPIO19I/O/TGPIO19I/O/T
27GPIO20U0RXDI1GPIO20I/O/T
28GPIO21U0TXDOGPIO21I/O/T
1
Boldmarks the default pin functions in the default boot mode. See Section3.1Chip
Boot Mode Control.
2
Regardinghighlightedcells, see Section2.3.3Restrictions for GPIOs.
3
Each IO MUX function (Fn,n= 0 ~ 2) is associated with atype. The description of
typeis as follows:
•I – input. O – output. T – high impedance.
•I1 – input; if the pin is assigned a function other than Fn, the input signal of
F
nis always1.
•I0 – input; if the pin is assigned a function other than Fn, the input signal of
F
nis always0.
Espressif Systems20
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
2.3.2Analog Functions
Some IO pins also haveanalog functions, for analog peripherals (such as ADC) in any power mode. Internal
analog signals are routed to these analog functions, see Table2-5Analog Signals Routed to Analog
Functions.
Table 2-5. Analog Signals Routed to Analog Functions
Pin FunctionSignalDescription
ADC..._CH...ADC1/2 channel ... signalADC1/2 interface
USB_D-Data -
USB Serial/JTAG function
USB_D+Data +
XTAL_32K_NNegative clock signal32 kHz external clock input/output
connected to ESP32-C3’s crystal or oscillatorXTAL_32K_PPositive clock signal
Table2-6Analog Functionsshows the analog functions of IO pins.
Table 2-6. Analog Functions
PinAnalog IOAnalog Function
2
No.Name
1, 2
F0F1
4GPIO0XTAL_32K_PADC1_CH0
5GPIO1XTAL_32K_NADC1_CH1
6GPIO2ADC1_CH2
8GPIO3ADC1_CH3
9GPIO4ADC1_CH4
10GPIO5ADC2_CH0
25GPIO18USB_D-
26GPIO19USB_D+
1
Boldmarks the default pin functions in the default
boot mode. See Section3.1ChipBootModeCon-
trol.
2
Regardinghighlightedcells, see Section2.3.3
Restrictions for GPIOs.
Espressif Systems21
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
2.3.3Restrictions for GPIOs
All IO pins of ESP32-C3 have GPIO pin functions. However, the IO pins are multiplexed and can be configured
for different purposes based on the requirements. Some IOs have restrictions for usage. It is essential to
consider the multiplexed nature and the limitations when using these IO pins.
In tables of this chapter, the following pin functions are highlighted inredoryellow. These functions
indicate pins that require extra caution when used asGPIO/GPIO:
•IO Pins– allocated for communication with in-package flash and NOT recommended for other uses.
For details, see Section2.6Pin Mapping Between Chip and Flash.
•IO Pins– have one of the following important functions:
–Strapping pins– need to be at certain logic levels at startup. See Section3Boot Configurations.
Note:
Strapping pins are highlighted byPin Nameor configurationsAt Reset, instead of the pin functions.
–USB_D+/-– by default, connected to the USB Serial/JTAG Controller. To function as GPIOs, these
pins need to be reconfigured.
–JTAG interface– often used for debugging. See Table2-3Peripheral Signals Routed via IO MUX.
To free these pins up, the pin functions USB_D+/- of the
ESP32-C3 Technical Reference Manual
USB Serial/JTAG Controllercan be used instead.
–UART0 interface– often used for debugging. See Table2-3Peripheral Signals Routed via IO MUX.
–VDD_SPI– the power supply pin for flash by default, and can only be used as a GPIO pin if the flash
is powered by an external power supply.
For more information about assigning pins, please see Section2.3.4Peripheral Pin AssignmentandESP32-C3
Consolidated Pin Overview
.
Espressif Systems22
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
2.3.4Peripheral Pin Assignment
Table2-7Peripheral Pin Assignmenthighlights which pins can be assigned to each peripheral interface
according to the following priorities:
•Priority 1: Fixed pins connected directly to peripheral signals via IO MUX.
If a peripheral interface does not have priority 1 pins, such as UART1, it can be assigned to any GPIO pins
from priority 2 to priority 4.
•Any GPIO pins mapping to peripheral signals via GPIO Matrix, can be priority 2, 3, or 4.
–Priority 2: GPIO pins can be freely used without restrictions.
–Priority 3: GPIO pins should be used with caution, as they may conflict with the following
important functions described in Section2.3.3Restrictions for GPIOs:
*GPIO2, GPIO8, GPIO9: Strapping pins.
*GPIO18, GPIO19: USB Serial/JTAG interface.
*GPIO4, GPIO5, GPIO6, GPIO7: JTAG interface.
*GPIO20, GPIO21: UART0 interface.
*GPIO11: The VDD_SPI pin. The power supply pin for flash by default, and can only be
reconfigured as a GPIO pin if the flash is powered by an external power supply.
–Priority 4: GPIO pins already allocated or not recommended for use, as described in Section2.3.3
Restrictions for GPIOs:
*GPIO12, GPIO13, GPIO14, GPIO15, GPIO16, GPIO17: SPI0/1 interface connected to the
in-package flash, or recommended for the off-package flash.
If a peripheral interface does not have priority 2 to 4 pins, such as USB Serial/JTAG, it means it can be
assigned only to priority 1 pins.
Note:
•For details about which peripheral signals are connected to IO MUX pins, please refer to Section2.3.1IO MUX
Functions.
•For details about which peripheral signals can be assigned to GPIO pins, please refer to
ESP32-C3 Technical Reference Manual&gt; Chapter IO MUX and GPIO Matrix &gt; Section Peripheral Signal List.
Espressif Systems23
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
Table 2-7. Peripheral Pin Assignment
Pin No.
Pin Name
USB Serial/JTAG
1
JTAG
ADC1
ADC2
UART0
2
SPI0/1
2
SPI2
2
UART1
I2C
I2S
TWAI
LED PWM
RMT
1
LNA_IN
2
VDD3P3
3
VDD3P3
4
XTAL_32K_P
ADC1_CH0 (P1)
GPIO0 (P2)
GPIO0 (P2)
GPIO0 (P2)
GPIO0 (P2)
GPIO0 (P2)
GPIO0 (P2)
GPIO0 (P2)
GPIO0 (P2)
GPIO0 (P2)
5
XTAL_32K_N
ADC1_CH1 (P1)
GPIO1 (P2)
GPIO1 (P2)
GPIO1 (P2)
GPIO1 (P2)
GPIO1 (P2)
GPIO1 (P2)
GPIO1 (P2)
GPIO1 (P2)
GPIO1 (P2)
6
GPIO2
ADC1_CH2 (P1)
GPIO2 (P3)
GPIO2 (P3)
FSPIQ (P1)
GPIO2 (P3)
GPIO2 (P3)
GPIO2 (P3)
GPIO2 (P3)
GPIO2 (P3)
GPIO2 (P3)
7
CHIP_EN
8
GPIO3
ADC1_CH3 (P1)
GPIO3 (P2)
GPIO3 (P2)
GPIO3 (P2)
GPIO3 (P2)
GPIO3 (P2)
GPIO3 (P2)
GPIO3 (P2)
GPIO3 (P2)
GPIO3 (P2)
9
MTMS
MTMS (P1)
ADC1_CH4 (P1)
GPIO4 (P3)
GPIO4 (P3)
FSPIHD (P1)
GPIO4 (P3)
GPIO4 (P3)
GPIO4 (P3)
GPIO4 (P3)
GPIO4 (P3)
GPIO4 (P3)
10
MTDI
MTDI (P1)
ADC2_CH0 (P1)
GPIO5 (P3)
GPIO5 (P3)
FSPIWP (P1)
GPIO5 (P3)
GPIO5 (P3)
GPIO5 (P3)
GPIO5 (P3)
GPIO5 (P3)
GPIO5 (P3)
11
VDD3P3_RTC
12
MTCK
MTCK (P1)
GPIO6 (P3)
GPIO6 (P3)
FSPICLK (P1)
GPIO6 (P3)
GPIO6 (P3)
GPIO6 (P3)
GPIO6 (P3)
GPIO6 (P3)
GPIO6 (P3)
13
MTDO
MTDO (P1)
GPIO7 (P3)
GPIO7 (P3)
FSPID (P1)
GPIO7 (P3)
GPIO7 (P3)
GPIO7 (P3)
GPIO7 (P3)
GPIO7 (P3)
GPIO7 (P3)
14
GPIO8
GPIO8 (P3)
GPIO8 (P3)
GPIO8 (P3)
GPIO8 (P3)
GPIO8 (P3)
GPIO8 (P3)
GPIO8 (P3)
GPIO8 (P3)
GPIO8 (P3)
15
GPIO9
GPIO9 (P3)
GPIO9 (P3)
GPIO9 (P3)
GPIO9 (P3)
GPIO9 (P3)
GPIO9 (P3)
GPIO9 (P3)
GPIO9 (P3)
GPIO9 (P3)
16
GPIO10
GPIO10 (P2)
GPIO10 (P2)
FSPICS0 (P1)
GPIO10 (P2)
GPIO10 (P2)
GPIO10 (P2)
GPIO10 (P2)
GPIO10 (P2)
GPIO10 (P2)
17
VDD3P3_CPU
18
VDD_SPI
GPIO11 (P3)
GPIO11 (P3)
GPIO11 (P3)
GPIO11 (P3)
GPIO11 (P3)
GPIO11 (P3)
GPIO11 (P3)
GPIO11 (P3)
GPIO11 (P3)
19
SPIHD
GPIO12 (P4)
SPIHD (P1)
GPIO12 (P4)
GPIO12 (P4)
GPIO12 (P4)
GPIO12 (P4)
GPIO12 (P4)
GPIO12 (P4)
GPIO12 (P4)
20
SPIWP
GPIO13 (P4)
SPIWP (P1)
GPIO13 (P4)
GPIO13 (P4)
GPIO13 (P4)
GPIO13 (P4)
GPIO13 (P4)
GPIO13 (P4)
GPIO13 (P4)
21
SPICS0
GPIO14 (P4)
SPICS0 (P1)
GPIO14 (P4)
GPIO14 (P4)
GPIO14 (P4)
GPIO14 (P4)
GPIO14 (P4)
GPIO14 (P4)
GPIO14 (P4)
22
SPICLK
GPIO15 (P4)
SPICLK (P1)
GPIO15 (P4)
GPIO15 (P4)
GPIO15 (P4)
GPIO15 (P4)
GPIO15 (P4)
GPIO15 (P4)
GPIO15 (P4)
23
SPID
GPIO16 (P4)
SPID (P1)
GPIO16 (P4)
GPIO16 (P4)
GPIO16 (P4)
GPIO16 (P4)
GPIO16 (P4)
GPIO16 (P4)
GPIO16 (P4)
24
SPIQ
GPIO17 (P4)
SPIQ (P1)
GPIO17 (P4)
GPIO17 (P4)
GPIO17 (P4)
GPIO17 (P4)
GPIO17 (P4)
GPIO17 (P4)
GPIO17 (P4)
25
GPIO18
USB_D- (P1)
GPIO18 (P3)
GPIO18 (P3)
GPIO18 (P3)
GPIO18 (P3)
GPIO18 (P3)
GPIO18 (P3)
GPIO18 (P3)
GPIO18 (P3)
GPIO18 (P3)
26
GPIO19
USB_D+ (P1)
GPIO19 (P3)
GPIO19 (P3)
GPIO19 (P3)
GPIO19 (P3)
GPIO19 (P3)
GPIO19 (P3)
GPIO19 (P3)
GPIO19 (P3)
GPIO19 (P3)
27
U0RXD
U0RXD (P1)
GPIO20 (P3)
GPIO20 (P3)
GPIO20 (P3)
GPIO20 (P3)
GPIO20 (P3)
GPIO20 (P3)
GPIO20 (P3)
GPIO20 (P3)
28
U0TXD
U0TXD (P1)
GPIO21 (P3)
GPIO21 (P3)
GPIO21 (P3)
GPIO21 (P3)
GPIO21 (P3)
GPIO21 (P3)
GPIO21 (P3)
GPIO21 (P3)
29
XTAL_N
30
XTAL_P
31
VDDA
32
VDDA
33
GND
1
For USB Serial/JTAG, the USB_D- and USB_D+ can be swapped by configuring the USB_SERIAL_JTAG_EXCHG_PINS bit according to
ESP32-C3 Technical Reference Manual
.
2
Signals of UART0, SPI0/1, and SPI2 interface can be mapped to any GPIO pins through the GPIO Matrix, regardless of whether they are directly routed to
fixed pins
via IO MUX.
3
For ESP32-C3FH4AZ and ESP32-C3FH4X, pins within the frame (namely pin 19
∼
pin 24) are not available.
Espressif Systems24
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
2.4Analog Pins
Table 2-8. Analog Pins
PinPinPinPin
No.NameTypeFunction
1LNA_INI/OLow Noise Amplifier (RF LNA) input / output signals
7CHIP_ENI
High: on, enables the chip (powered up).
Low: off, disables the chip (powered down).
Note: Do not leave the CHIP_EN pin floating.
29XTAL_N—External clock input/output connected to the chip’s crystal or
oscillator. P/N means differential clock positive/negative.30XTAL_P—
Espressif Systems25
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
2.5Power Supply
2.5.1Power Pins
The chip is powered via the power pins described in Table2-9Power Pins.
Table 2-9. Power Pins
PinPinPower Supply
1,2
No.NameDirectionPower Domain / OtherIO Pins
3
2VDD3P3InputAnalog power domain
3VDD3P3InputAnalog power domain
11VDD3P3_RTCInputRTC and part of Digital power domainsRTC IO
17VDD3P3_CPUInputDigital power domainDigital IO
18VDD_SPI
InputIn-package flash (backup power line)
OutputIn-package and off-package flashSPI IO
31VDDAInputAnalog power domain
32VDDAInputAnalog power domain
33GND—External ground connection
1
See in conjunction with Section2.5.2Power Scheme.
2
For recommended and maximum voltage and current, see Section5.1Absolute Max-
imum Ratingsand Section5.2Recommended Operating Conditions.
3
Digital IO pins are those powered by VDD3P3_CPU, and RTC IO pins are those pow-
ered by VDD3P3_RTC and so on, as shown in Figure2-3ESP32-C3 Power Scheme.
See also Table2-1Pin Overview&gt; ColumnPin Providing Power.
2.5.2Power Scheme
The power scheme is shown in Figure2-3ESP32-C3 Power Scheme.
The components on the chip are powered via voltage regulators.
Table 2-10. Voltage Regulators
Voltage RegulatorOutputPower Supply
Digital1.1 VDigital power domain
Low-power1.1 VRTC power domain
Espressif Systems26
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
Figure 2-3. ESP32-C3 Power Scheme
2.5.3Chip Power-up and Reset
Once the power is supplied to the chip, its power rails need a short time to stabilize. After that, CHIP_EN – the
pin used for power-up and reset – is pulled high to activate the chip. For information on CHIP_EN as well as
power-up and reset timing, see Figure
2-4and Table2-11.
V
IL_nRST
t
STBL
t
RST
2.8 V
VDDA,
VDD3P3,
VDD3P3_RTC,
VDD3P3_CPU
CHIP_EN
Figure 2-4. Visualization of Timing Parameters for Power-up and Reset
Espressif Systems27
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
Table 2-11. Description of Timing Parameters for Power-up and Reset
ParameterDescriptionMin (μs)
t
ST BL
Time  reserved  for  the  power  rails  of  VDDA,  VDD3P3,
VDD3P3_RTC, and VDD3P3_CPU to stabilize before the CHIP_EN
pin is pulled high to activate the chip
50
t
RST
Time reserved for CHIP_EN to stay below V
IL_nRST
to reset the
chip (see Table5-4)
50
Espressif Systems28
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

2  Pins
2.6Pin Mapping Between Chip and Flash
Table2-12lists the pin mapping between the chip and flash for all SPI modes.
For chip variants with in-package flash (see Table1-1ESP32-C3 Series Comparison), the pins allocated for
communication with in-package flash can be identified depending on the SPI mode used.
For off-package flash, these are the recommended pin mappings.
For more information on SPI controllers, see also Section4.2.1.2SPI Controller.
Notice:
Do not use the pins connected to in-package flash for any other purposes.
Table 2-12. Pin Mapping Between Chip and In-package Flash
PinPinSingle SPIDual SPIQuad SPI / QPI
No.NameFlashFlashFlash
22SPICLKCLKCLKCLK
21SPICS0
1
CS#CS#CS#
23SPIDDIDIDI
24SPIQDODODO
20SPIWPWP#WP#WP#
19SPIHDHOLD#HOLD#HOLD#
1
CS0 is for in-package flash
Espressif Systems29
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

3  Boot Configurations
3Boot Configurations
The chip allows for configuring the following boot parameters throughstrapping pinsandeFuse parametersat
power-up or a hardware reset, without microcontroller interaction.
•Chip boot mode
–Strapping pins: GPIO2, GPIO8, and GPIO9
•ROM message printing
–Strapping pin: GPIO8
–eFuse parameters: EFUSE_UART_PRINT_CONTROL and EFUSE_USB_PRINT_CHANNEL
The default values of all the above eFuse parameters are 0, which means that they are not burnt. Given that
eFuse is one-time programmable, once programmed to 1, it can never be reverted to 0. For how to program
eFuse parameters, please refer toESP32-C3 Technical Reference Manual&gt; ChaptereFuse Controller.
The default values of the strapping pins, namely the logic levels, are determined by pins’ internal weak
pull-up/pull-down resistors at reset if the pins are not connected to any circuit, or connected to an external
high-impedance circuit.
Table 3-1. Default Configuration of Strapping Pins
Strapping PinDefault ConfigurationBit Value
GPIO2Floating–
GPIO8Floating–
GPIO9Weak pull-up1
To change the bit values, the strapping pins should be connected to external pull-down/pull-up resistances. If
the ESP32-C3 is used as a device by a host MCU, the strapping pin voltage levels can also be controlled by
the host MCU.
All strapping pins have latches. At Chip Reset, the latches sample the bit values of their respective strapping
pins and store them until the chip is powered down or shut down. The states of latches cannot be changed in
any other way. It makes the strapping pin values available during the entire chip operation, and the pins are
freed up to be used as regular IO pins after reset. For details on Chip Reset, see
ESP32-C3 Technical Reference Manual&gt; ChapterReset and Clock.
The timing of signals connected to the strapping pins should adhere to thesetup timeandhold time
specifications in Table
3-2and Figure3-1.
Table 3-2. Description of Timing Parameters for the Strapping Pins
ParameterDescriptionMin (ms)
t
SU
Setup timeis the time reserved for the power rails to stabilize be-
fore the CHIP_EN pin is pulled high to activate the chip.
0
t
H
Hold timeis the time reserved for the chip to read the strapping
pin values after CHIP_EN is already high and before these pins
start operating as regular IO pins.
3
Espressif Systems30
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

3  Boot Configurations
Strapping pin
V
IH_nRST
V
IH
t
SU
t
H
CHIP_EN
Figure 3-1. Visualization of Timing Parameters for the Strapping Pins
3.1Chip Boot Mode Control
GPIO2, GPIO8, and GPIO9 control the boot mode after the reset is released. See Table3-3Chip Boot Mode
Control
.
Table 3-3. Chip Boot Mode Control
Boot ModeGPIO2
2
GPIO8GPIO9
SPI boot mode1Any value1
Joint download boot mode
3
110
1
Boldmarks the default value and configuration.
2
GPIO2 actually does not determine SPI Boot and Joint Down-
load Boot mode, but it is recommended to pull this pin up due
to glitches.
3
Joint Download Boot mode supports the following download
methods:
•USB-Serial-JTAG Download Boot
•UART Download Boot
In SPI Boot mode, the ROM bootloader loads and executes the program from SPI flash to boot the
system.
In Joint Download Boot mode, users can download binary files into flash using UART0 or USB interface. It is
also possible to download binary files into SRAM and execute it from SRAM.
In addition to SPI Boot and Joint Download Boot modes, ESP32-C3 also supports SPI Download Boot mode.
For details, please see
ESP32-C3 Technical Reference Manual&gt; ChapterChip Boot Control.
3.2ROM Messages Printing Control
During the boot process, the messages by the ROM code can be printed to:
•(Default) UART0 and USB Serial/JTAG controller
Espressif Systems31
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

3  Boot Configurations
•UART0
•USB Serial/JTAG controller
EFUSE_UART_PRINT_CONTROL and GPIO8 control ROM messages printing toUART0as shown in Table3-4
UART0 ROM Message Printing Control.
Table 3-4. UART0 ROM Message Printing Control
UART0 ROM Code PrintingEFUSE_UART_PRINT_CONTROLGPIO8
Enabled
0Ignored
10
21
Disabled
11
20
3Ignored
1
Boldmarks the default value and configuration.
EFUSE_USB_PRINT_CHANNEL controls the printing toUSB Serial/JTAG controlleras shown in Table3-5USB
Serial/JTAG ROM Message Printing Control.
Table 3-5. USB Serial/JTAG ROM Message Printing Control
USB Serial/JTAG
ROM Code Printing
EFUSE_DIS_USB_SERIAL_JTAG
2
EFUSE_USB_PRINT_CHANNEL
Enabled00
Disabled
01
1Ignored
1
Boldmarks the default value and configuration.
2
EFUSE_DIS_USB_SERIAL_JTAG controls whether to disable USB Serial/JTAG.
Espressif Systems32
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
4Functional Description
4.1System
This section describes the core of the chip’s operation, covering its microprocessor, memory organization,
system components, and security features.
4.1.1Microprocessor and Master
This subsection describes the core processing units within the chip and their capabilities.
4.1.1.1High-Performance CPU
ESP32-C3 has a low-power 32-bit RISC-V single-core microprocessor with the following features:
•Four-stage pipeline that supports a clock frequency of up to 160 MHz
•RV32IMC ISA
•32-bit multiplier and 32-bit divider
•Up to 32 vectored interrupts at seven priority levels
•Up to 8 hardware breakpoints/watchpoints
•Up to 16 PMP regions
•JTAG for debugging
For details, see
ESP32-C3 Technical Reference Manual&gt; ChapterHigh-Performance CPU.
4.1.1.2GDMA Controller
ESP32-C3 has a general DMA controller (GDMA) with six independent channels, i.e. three transmit channels
and three receive channels. These six channels are shared by peripherals with DMA feature. The GDMA
controller implements a fixed-priority scheme among these channels, whose priority can be configured.
The GDMA controller controls data transfer using linked lists. It allows peripheral-to-memory and
memory-to-memory data transfer at a high speed. All channels can access internal RAM.
Peripherals on ESP32-C3 with DMA feature are SPI2, UHCI0, I2S, AES, SHA, and ADC.
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterGDMA Controller (DMA).
4.1.2Memory Organization
This subsection describes the memory arrangement to explain how data is stored, accessed, and managed
for efficient operation.
Figure4-1illustrates the address mapping structure of ESP32-C3.
Espressif Systems33
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
Figure 4-1. Address Mapping Structure
Note:
The memory space with gray background is not available for use.
4.1.2.1Internal Memory
The internal memory of ESP32-C3 refers to the memory integrated on the chip die or in the chip package,
including ROM, SRAM, eFuse, and flash.
•384 KB of ROM: for booting and core functions
•400 KB of on-chip SRAM: for data and instructions, running at a configurable frequency of up to 160
MHz. Of the 400 KB SRAM, 16 KB is configured for cache
•RTC FAST memory: 8 KB of SRAM that can be accessed by the main CPU. It can retain data in
Deep-sleep mode
•4 Kbit of eFuse: 1792 bits are reserved for your data, such as encryption key and device ID
Espressif Systems34
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
•In-package flash
–See flash size in Chapter1ESP32-C3 Series Comparison
–For specifications, refer to Section5.7Memory Specifications.
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterSystem and Memory.
4.1.2.2External Memory
ESP32-C3 allows connection to memories outside the chip’s package via the SPI, Dual SPI, Quad SPI, and QPI
interfaces.
CPU’s instruction memory space and read-only data memory space can map into the off-package flash of
ESP32-C3, whose size can be 16 MB at most. ESP32-C3 supports hardware encryption/decryption based on
XTS-AES to protect developers’ programs and data in flash.
Through high-speed caches, ESP32-C3 can support at a time up to:
•8 MB of instruction memory space which can map into flash as individual blocks of 64 KB. 8-bit, 16-bit
and 32-bit reads are supported.
•8 MB of data memory space which can map into flash as individual blocks of 64 KB. 8-bit, 16-bit and
32-bit reads are supported.
Note:
After ESP32-C3 is initialized, software can customize the mapping of off-package flash into the CPU address space.
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterSystem and Memory.
4.1.2.3Cache
ESP32-C3 has an eight-way set associative cache. This cache is read-only and has the following
features:
•Size: 16 KB
•Block size: 32 bytes
•Pre-load function
•Lock function
•Critical word first and early restart
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterSystem and Memory.
4.1.2.4eFuse Controller
TheeFusememory is a one-time programmable memory that stores parameters and user data, and the eFuse
controller of ESP32-C3 is used to program and read this eFuse memory.
Feature List
•Configurable write protection
Espressif Systems35
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
•Configurable read protection
•Various hardware encoding schemes against data corruption
For details, seeESP32-C3 Technical Reference Manual&gt; ChaptereFuse Controller.
4.1.3System Components
This subsection describes the essential components that contribute to the overall functionality and control of
the system.
4.1.3.1IO MUX and GPIO Matrix
ESP32-C3 has 22 or 16 GPIO pins which can be assigned various functions by configuring corresponding
registers. Besides digital signals, some GPIOs can be also used for analog functions, such as ADC.
All GPIOs have selectable internal pull-up or pull-down, or can be set to high impedance. When these GPIOs
are configured as an input, the input value can be read by software through the register. Input GPIOs can also
be set to generate edge-triggered or level-triggered CPU interrupts. All digital IO pins are bi-directional,
non-inverting and tristate, including input and output buffers with tristate control. These pins can be
multiplexed with other functions, such as the UART, SPI, etc. For low-power operations, the GPIOs can be set
to holding state.
The IO MUX and the GPIO matrix are used to route signals from peripherals to GPIO pins. Together they
provide highly configurable I/O. Using GPIO Matrix, peripheral input signals can be configured from any IO pins
while peripheral output signals can be configured to any IO pins.
For details, see
ESP32-C3 Technical Reference Manual&gt; ChapterIO MUX and GPIO Matrix.
4.1.3.2Reset
The ESP32-C3 chip provides four types of reset that occur at different levels, namely CPU Reset, Core Reset,
System Reset, and Chip Reset. Except for Chip Reset, all reset types preserve the data stored in internal
memory.
Feature List
•Support four reset levels:
–CPU Reset: Only resets CPU core. Once such reset is released, the instructions from the CPU reset
vector will be executed
–Core Reset: Resets the whole digital system except RTC, including CPU, peripherals, Wi-Fi,
Bluetooth
®
LE, and digital GPIOs
–System Reset: Resets the whole digital system, including RTC
–Chip Reset: Resets the whole chip
•Support software reset and hardware reset:
–Software Reset: The CPU can trigger a software reset by configuring the corresponding registers
–Hardware Reset: Hardware reset is directly triggered by the circuit
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterReset and Clock.
Espressif Systems36
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
4.1.3.3Clock
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterReset and Clock.
CPU Clock
The CPU clock has three possible sources:
•External main crystal clock
•Fast RC oscillator (typically about 17.5 MHz, and adjustable)
•PLL clock
The application can select the clock source from the three clocks above. The selected clock source drives
the CPU clock directly, or after division, depending on the application. Once the CPU is reset, the default
clock source would be the external main crystal clock divided by 2.
Note:
ESP32-C3 is unable to operate without an external main crystal clock.
RTC Clock
The RTC slow clock is used for RTC counter, RTC watchdog and low-power controller. It has three possible
sources:
•External low-speed (32 kHz) crystal clock
•Internal slow RC oscillator (typically about 136 kHz, and adjustable)
•Internal fast RC oscillator divided clock (derived from the fast RC oscillator divided by 256)
The RTC fast clock is used for RTC peripherals and sensor controllers. It has two possible sources:
•External main crystal clock divided by 2
•Internal fast RC oscillator divide-by-N clock (typically about 17.5 MHz, and adjustable)
4.1.3.4Interrupt Matrix
The Interrupt Matrix in the ESP32-C3 chip independently routes peripheral interrupt sources to the ESP-RISC-V
CPU’s peripheral interrupts, to timely inform CPU to process the coming interrupts.
Feature List
•Accept 62 peripheral interrupt sources as input
•Generate 31 CPU peripheral interrupts to CPU as output
•Query current interrupt status of peripheral interrupt sources
•Configure priority, type, threshold, and enable signal of CPU interrupts
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterInterrupt Matrix.
Espressif Systems37
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
4.1.3.5System Timer
ESP32-C3 integrates a 52-bit system timer, which has two 52-bit counters and three comparators. The system
timer has the following features:
•Counters with a fixed clock frequency of 16 MHz
•Three types of independent interrupts generated according to alarm value
•Two alarm modes: target mode and period mode
•52-bit target alarm value and 26-bit periodic alarm value
•Automatic reload of counter value
•Counters can be stalled if the CPU is stalled or in OCD mode
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterSystem Timer.
4.1.3.6Power Management Unit
The ESP32-C3 has an advanced Power Management Unit (PMU). It can be flexibly configured to power up
different power domains of the chip to achieve the best balance between chip performance, power
consumption, and wakeup latency.
Configuring the PMU is a complex procedure. To simplify power management for typical scenarios, there are
the followingpredefined power modesthat power up different combinations of power domains:
•Active mode– The CPU, RF circuits, and all peripherals are on. The chip can process data, receive,
transmit, and listen.
•Modem-sleep mode– The CPU is on, but the clock frequency can be reduced. The wireless
connections can be configured to remain active as RF circuits are periodically switched on when
required.
•Light-sleep mode– The CPU stops running, and can be optionally powered on. The chip can be woken
up via all wake up mechanisms: MAC, RTC timer, or external interrupts. Wireless connections can remain
active. Some groups of digital peripherals can be optionally shut down.
•Deep-sleep mode– Only RTC is powered on. Wireless connection data is stored in RTC memory.
For power consumption in different power modes, see Section5.6Current Consumption.
Figure4-2Components and Power Domainsand the following Table4-1show the distribution of chip
components between
power domainsandpower subdomains.
Espressif Systems38
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
Wireless Digital Circuits
Wi-Fi MAC 
Wi-Fi 
Baseband
Bluetooth LE Link 
Controller
Bluetooth LE 
Baseband
Digital Power Domain
CPU
RISC-V
32-bit
Microprocessor
JTAG
Cache
Espressif’s ESP32-C3 Wi-Fi + Bluetooth
®
 Low Energy SoC
ROMSRAM
Optional Digital Peripherals 
RSADigital SignatureSHA
AES
HMAC
Secure BootSPI2GDMA
2.4 GHz Balun 
+ Switch
2.4 GHz 
Receiver
2.4 GHz 
Transmitter
RF 
Synthesizer
RF Circuits
Phase Lock 
Loop
PLL
XTAL_CLK
External Main 
Clock
RC_FAST_CLK
Fast RC 
Oscillator
Analog Power Domain
Flash 
Encryption
RNG
USB Serial/
JTAG
GPIO
UART
TWAI
®
General-
purpose 
Timers
I2S
I2C
LED PWM
SPI0/1
RMT
DIG ADC
System 
Timer
Main System 
Watchdog 
Timers
Power distribution
Power domain
Power subdomain
RTC Memory
RTC 
Watchdog 
Timer
PMU
RTC Power Domain
eFuse 
Controller
Brownout 
Detector
Super 
Watchdog
World 
Controller
Debug 
Assistant
RTC Timer
Temperature 
Sensor
Figure 4-2. Components and Power Domains
Table 4-1. Components and Power Domains
RTCDigitalAnalog
Power
Mode
Power
Domain
CPU
Optional
Digital
Periph
Wireless
Digital
Circuits
FOSC_
CLK
XTAL_
CLK
PLL
RF
Circuits
ActiveONONONONONONONONONON
Modem-sleepONONONONON
1
ONONONONOFF
2
Light-sleepONONOFF
1
ON
1
OFF
1
ONOFFOFFOFFOFF
2
Deep-sleepONOFFOFFOFFOFFONOFFOFFOFFOFF
1
Configurable, see the TRM.
2
If Wireless Digital Circuits are on, RF circuits are periodically switched on when required by internal operation
to keep active wireless connections running.
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterLow-Power Management (RTC_CNTL).
Espressif Systems39
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
4.1.3.7Timer Group
ESP32-C3 has two 54-bit general-purpose timers, which are based on 16-bit prescalers and 54-bit
auto-reload-capable up/down-timers.
The timers’ features are summarized as follows:
•A 16-bit clock prescaler, from 1 to 65536
•A 54-bit time-base counter programmable to be incrementing or decrementing
•Able to read real-time value of the time-base counter
•Halting and resuming the time-base counter
•Programmable alarm generation
•Level interrupt generation
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterTimer Group (TIMG).
4.1.3.8Watchdog Timers
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterWatchdog Timers.
Digital Watchdog Timers
ESP32-C3 contains three digital watchdog timers: one in each of the two timer groups (called Main System
Watchdog Timers, or MWDT) and one in the RTC module (called the RTC Watchdog Timer, or RWDT).
During the flash boot process, RWDT and the MWDT in timer group 0 (TIMG0) are enabled automatically in
order to detect and recover from booting errors.
Digital watchdog timers have the following features:
•Four stages, each with a programmable timeout value. Each stage can be configured, enabled and
disabled separately
•Interrupt, CPU reset, or core reset for MWDT upon expiry of each stage; interrupt, CPU reset, core reset,
or system reset for RWDT upon expiry of each stage
•32-bit expiry counter
•Write protection, to prevent RWDT and MWDT configuration from being altered inadvertently
•Flash boot protection
If the boot process from an SPI flash does not complete within a predetermined period of time, the
watchdog will reboot the entire main system.
Analog Watchdog Timer
ESP32-C3 also has one analog watchdog timer: RTC super watchdog timer (SWD). It is an ultra-low-power
circuit in analog domain that helps to prevent the system from operating in a sub-optimal state and resets the
system if required.
SWD has the following features:
•Ultra-low power
Espressif Systems40
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
•Interrupt to indicate that the SWD timeout period is close to expiring
•Various dedicated methods for software to feed SWD, which enables SWD to monitor the working state
of the whole operating system
4.1.3.9XTAL32K Watchdog Timers (XTWDT)
The XTAL32K watchdog timer on ESP32-C3 monitors the status of the 32 kHz external crystal. When detecting
the oscillation failure of this clock, it changes the slow clock source of RTC. The clock source frequency is
configurable.
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterXTAL32K Watchdog Timers (XTWDT).
4.1.3.10Permission Control
ESP32-C3 includes a Permission Controller (PMS), which allocates the hardware resources (memory and
peripherals) to two isolated environments, thereby realizing the separation of privileged and unprivileged
environments.
Feature List
•Independent access management in a privileged environment and unprivileged environment
•Independent access management to internal memory, including
–CPU access to internal memory
–GDMA access to internal memory
•Independent access management to external memory, including
–CPU to external memory via SPI1
–CPU to external memory via Cache
•Independent access management to peripheral regions, including
–CPU access to peripheral regions
–Interrupt upon unsupported access alignment
•Address splitting for more flexible access management
•Register locks to secure the integrity of access management related registers
•Interrupt upon unauthorized access
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterPermission Control (PMS).
4.1.3.11World Controller (WCL)
The hardware and software resources on ESP32-C3 can be partitioned into Secure World and Non-secure
World. Transitions between these worlds are controlled and logged by the World Controller.
Espressif Systems41
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
Feature List
•Secure World (World0):
–Can access all peripherals and memories;
–Performs all security related operations, such user authentication, secure communication, and data
encryption and decryption, etc.
•Non-secure World (World1):
–Can access some peripherals and memories;
–Performs other operations, such as user operation and different applications, etc.
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterWorld Controller (WCL).
4.1.3.12System Registers
The System Registers in the ESP32-C3 chip are used to configure various auxiliary chip features.
Feature List
•Control system and memory
•Control clock
•Control software interrupt
•Control low-power management
•Control peripheral clock gating and reset
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterSystem Registers (HP_SYSREG).
4.1.3.13Debug Assistant
The Debug Assistant provides a set of functions to help locate bugs and issues during software debugging. It
offers various monitoring capabilities and logging features to assist in identifying and resolving software errors
efficiently.
Feature List
•Read/write monitoring: Monitors whether the CPU bus has read from or written to a specified address
space. A detected read or write will trigger an interrupt.
•Stack pointer (SP) monitoring: Monitors whether the SP exceeds the specified address space. A
bounds violation will trigger an interrupt.
•Program counter (PC) logging: Records PC value. The developer can get the last PC value at the most
recent CPU reset.
•Bus access logging: Records the information about bus access. When the CPU or DMA writes a
specified value, the Debug Assistant module will record the address and PC value of this write operation,
and push the data to the SRAM.
For details, see
ESP32-C3 Technical Reference Manual&gt; ChapterDebug Assistant (ASSIST_DEBUG).
Espressif Systems42
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
4.1.4Cryptography and Security Component
This subsection describes the security features incorporated into the chip, which safeguard data and
operations.
4.1.4.1AES Accelerator
ESP32-C3 integrates an Advanced Encryption Standard (AES) accelerator, which is a hardware device that
speeds up computation using AES algorithm significantly, compared to AES algorithms implemented solely in
software. The AES accelerator integrated in ESP32-C3 has two working modes, which are Typical AES and
DMA-AES.
Feature List
•Typical AES working mode
–AES-128/AES-256 encryption and decryption
•DMA-AES working mode
–AES-128/AES-256 encryption and decryption
–Block cipher mode
*ECB (Electronic Codebook)
*CBC (Cipher Block Chaining)
*OFB (Output Feedback)
*CTR (Counter)
*CFB8 (8-bit Cipher Feedback)
*CFB128 (128-bit Cipher Feedback)
–Interrupt on completion of computation
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterAES Accelerator (AES).
4.1.4.2HMAC Accelerator
The HMAC Accelerator (HMAC) module is designed to compute Message Authentication Codes (MACs) using
the SHA-256 Hash algorithm and keys as described in RFC 2104. It provides hardware support for HMAC
computations, significantly reducing software complexity and improving performance.
Feature List
•Standard HMAC-SHA-256 algorithm
•Hash result only accessible by configurable hardware peripheral (in downstream mode)
•Compatible to challenge-response authentication algorithm
•Generates required keys for the Digital Signature (DS) peripheral (in downstream mode)
•Re-enables soft-disabled JTAG (in downstream mode)
Espressif Systems43
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
For details, see theESP32-C3 Technical Reference Manual&gt; ChapterHMAC Accelerator.
4.1.4.3RSA Accelerator
The RSA accelerator provides hardware support for high-precision computation used in various RSA
asymmetric cipher algorithms, significantly improving their run time and reducing their software complexity.
Compared with RSA algorithms implemented solely in software, this hardware accelerator can speed up RSA
algorithms significantly.
Feature List
•Large-number modular exponentiation with two optional acceleration options, operands width up to
3072 bits
•Large-number modular multiplication, operands width up to 3072 bits
•Large-number multiplication, operands width up to 1536 bits
•Operands of different widths
•Interrupt on completion of computation
For details, see theESP32-C3 Technical Reference Manual&gt; ChapterRSA Accelerator.
4.1.4.4SHA Accelerator
The SHA Accelerator (SHA) is a hardware device that speeds up SHA algorithm significantly, compared to SHA
algorithm implemented solely in software. The SHA accelerator integrated in ESP32-C3 has two working
modes, which are Typical SHA and DMA-SHA.
Feature List
•The following hash algorithms introduced inFIPSPUB180-4Spec.
–SHA-1
–SHA-224
–SHA-256
•Two working modes
–Typical SHA
–DMA-SHA
•Interleaved function when working in Typical SHA working mode
•Interrupt function when working in DMA-SHA working mode
For more details, see the
ESP32-C3 Technical Reference Manual&gt; ChapterSHA Accelerator (SHA).
4.1.4.5Digital Signature
The Digital Signature (DS) module in the ESP32-C3 chip generates message signatures based on RSA with
hardware acceleration.
Espressif Systems44
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
Feature List
•RSA digital signatures with key length up to 3072 bits
•Encrypted private key data, only decryptable by DS module
•SHA-256 digest to protect private key data against tampering by an attacker
For more details, see theESP32-C3 Technical Reference Manual&gt; Chapter
Digital Signature (DS)
.
4.1.4.6External Memory Encryption and Decryption
The External Memory Encryption and Decryption (XTS_AES) module in the ESP32-C3 chip provides security
for users’ application code and data stored in the external memory (flash).
Feature List
•General XTS_AES algorithm, compliant with IEEE Std 1619-2007
•Software-based manual encryption
•High-speed auto decryption, without software’s participation
•Encryption and decryption functions jointly determined by registers configuration, eFuse parameters,
and boot mode
For more details, see theESP32-C3 Technical Reference Manual&gt; ChapterExternal Memory Encryption and
Decryption (XTS_AES).
4.1.4.7Random Number Generator
The Random Number Generator (RNG) in the ESP32-C3 is a true random number generator that generates
32-bit random numbers for cryptographic operations from a physical process.
Feature List
•RNG entropy source
–Thermal noise from high-speed ADC or SAR ADC
–An asynchronous clock mismatch
For more details about the Random Number Generator, refer to theESP32-C3 Technical Reference Manual&gt;
ChapterRandom Number Generator (RNG).
4.1.4.8Clock Glitch Detection
The Clock Glitch Detection module on ESP32-C3 detects glitches in external crystal clock signals, and
generates a system reset signal when detecting glitches to reset the whole digital circuit including RTC.
For more details about the Random Number Generator, refer to theESP32-C3 Technical Reference Manual&gt;
ChapterClock Glitch Detection.
Espressif Systems45
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
4.2Peripherals
This section describes the chip’s peripheral capabilities, covering connectivity interfaces and on-chip sensors
that extend its functionality.
4.2.1Connectivity Interface
This subsection describes the connectivity interfaces on the chip that enable communication and interaction
with external devices and networks.
4.2.1.1UART Controller
ESP32-C3 has two UART interfaces, i.e. UART0 and UART1, which support IrDA and asynchronous
communication (RS232 and RS485) at a speed of up to 5 Mbps. The UART controller provides hardware flow
control (CTS and RTS signals) and software flow control (XON and XOFF). Both UART interfaces connect to
GDMA via UHCI0, and can be accessed by the GDMA controller or directly by the CPU.
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterUART Controller (UART, LP_UART).
Pin Assignment
For details, see Section2.3.4Peripheral Pin Assignment.
4.2.1.2SPI Controller
ESP32-C3 has the following SPI interfaces:
•SPI0used by ESP32-C3’s GDMA controller and cache to access in-package or off-package flash
•SPI1used by the CPU to access in-package or off-package flash
•SPI2is a general purpose SPI controller with access to a DMA channel allocated by the GDMA controller
Features of SPI0 and SPI1
•Supports Single SPI, Dual SPI, and Quad SPI, QPI modes
•Configurable clock frequency with a maximum of 120 MHz in Single Transfer Rate (STR) mode
•Data transmission is in bytes
Features of SPI2
•Supports operation as a master or slave
•Connects to a DMA channel allocated by the GDMA controller
•Supports Single SPI, Dual SPI, and Quad SPI, QPI
•Configurable clock polarity (CPOL) and phase (CPHA)
•Configurable clock frequency
•Data transmission is in bytes
Espressif Systems46
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
•Configurable read and write data bit order: most-significant bit (MSB) first, or least-significant bit (LSB)
first
•As a master
–Supports 2-line full-duplex communication with clock frequency up to 80 MHz
–Supports 1-, 2-, 4-line half-duplex communication with clock frequency up to 80 MHz
–Provides six SPI_CS pins for connection with six independent SPI slaves
–Configurable CS setup time and hold time
•As a slave
–Supports 2-line full-duplex communication with clock frequency up to 60 MHz
–Supports 1-, 2-, 4-line half-duplex communication with clock frequency up to 60 MHz
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterSPI Controller (SPI).
Pin Assignment
For details, see Section2.3.4Peripheral Pin Assignment.
4.2.1.3I2C Controller
ESP32-C3 has an I2C bus interface which is used for I2C master mode or slave mode, depending on your
configuration. The I2C interface supports:
•Standard mode (100 Kbit/s)
•Fast mode (400 Kbit/s)
•Up to 800 Kbit/s (constrained by SCL and SDA pull-up strength)
•7-bit and 10-bit addressing mode
•Double addressing mode
•7-bit broadcast address
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterI2C Controller (I2C).
Pin Assignment
For details, see Section2.3.4Peripheral Pin Assignment.
4.2.1.4I2S Controller
ESP32-C3 includes a standard I2S interface. This interface can operate as a master or a slave in full-duplex
mode or half-duplex mode, and can be configured for 8-bit, 16-bit, 24-bit, or 32-bit serial communication. BCK
clock frequency, from 10 kHz up to 40 MHz, is supported.
The I2S interface connects to the GDMA controller. The interface supports TDM PCM, TDM MSB alignment,
TDM standard, and PDM standard.
For details, see
ESP32-C3 Technical Reference Manual&gt; ChapterI2S Controller (I2S).
Espressif Systems47
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
Pin Assignment
For details, see Section2.3.4Peripheral Pin Assignment.
4.2.1.5USB Serial/JTAG Controller
ESP32-C3 integrates a USB Serial/JTAG controller. This controller has the following features:
•CDC-ACM virtual serial port and JTAG adapter functionality
•USB 2.0 full speed compliant, capable of up to 12 Mbit/s transfer speed (Note that this controller does
not support the faster 480 Mbit/s high-speed transfer mode)
•Programming in-package/off-package flash
•CPU debugging with compact JTAG instructions
•A full-speed USB PHY integrated in the chip
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterUSB Serial/JTAG Controller
(USB_SERIAL_JTAG).
Pin Assignment
For details, see Section2.3.4Peripheral Pin Assignment.
4.2.1.6Two-wire Automotive Interface
ESP32-C3 has a TWAI
®
controller with the following features:
•Compatible with ISO 11898-1 protocol (CAN Specification 2.0)
•Standard frame format (11-bit ID) and extended frame format (29-bit ID)
•Bit rates from 1 Kbit/s to 1 Mbit/s
•Multiple modes of operation: Normal, Listen Only, and Self-Test (no acknowledgment required)
•64-byte receive FIFO
•Acceptance filter (single and dual filter modes)
•Error detection and handling: error counters, configurable error interrupt threshold, error code capture,
arbitration lost capture
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterTwo-wire Automotive Interface.
Pin Assignment
For details, see Section2.3.4Peripheral Pin Assignment.
4.2.1.7LED PWM Controller
The LED PWM controller can generate independent digital waveform on six channels. The LED PWM
controller:
Espressif Systems48
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
•Can generate digital waveform with configurable periods and duty cycle. The resolution of duty cycle
can be up to 14 bits.
•Has multiple clock sources, including APB clock and external main crystal clock.
•Can operate when the CPU is in Light-sleep mode.
•Supports gradual increase or decrease of duty cycle, which is useful for the LED RGB color-gradient
generator.
For details, seeESP32-C3 Technical Reference Manual&gt; ChapterLED PWM Controller.
Pin Assignment
For details, see Section2.3.4Peripheral Pin Assignment.
4.2.1.8Remote Control Peripheral
The Remote Control Peripheral (RMT) supports two channels of infrared remote transmission and two
channels of infrared remote reception. By controlling pulse waveform through software, it supports various
infrared and other single wire protocols. All four channels share a 192 × 32-bit memory block to store transmit
or receive waveform.
For more details, seeESP32-C3 Technical Reference Manual&gt; ChapterRemote Control Peripheral (RMT).
Pin Assignment
For details, see Section2.3.4Peripheral Pin Assignment.
4.2.2Analog Signal Processing
This subsection describes components on the chip that sense and process real-world data.
4.2.2.1SAR ADC
ESP32-C3 integrates two 12-bit SAR ADCs.
•ADC1 supports measurements on 5 channels, and is factory-calibrated.
•ADC2 supports measurements on 1 channel, and is not factory-calibrated.
Note:
ADC2 of some chip revisions is not operable. For details, please refer toESP32-C3SeriesSoCErrata.
For ADC characteristics, please refer to Section5.5ADC Characteristics.
For more details, seeESP32-C3 Technical Reference Manual&gt; ChapterOn-Chip Sensors and Analog Signal
Processing.
Pin Assignment
For details, see Section2.3.4Peripheral Pin Assignment.
Espressif Systems49
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
4.2.2.2Temperature Sensor
The temperature sensor generates a voltage that varies with temperature. The voltage is internally converted
via an ADC into a digital value.
The temperature sensor has a range of40 °C to 125 °C. It is designed primarily to sense the temperature
changes inside the chip. The temperature value depends on factors like microcontroller clock frequency or
I/O load. Generally, the chip’s internal temperature is higher than the operating ambient temperature.
For more details, seeESP32-C3 Technical Reference Manual&gt; ChapterOn-Chip Sensors and Analog Signal
Processing.
Espressif Systems50
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
4.3Wireless Communication
This section describes the chip’s wireless communication capabilities, spanning radio technology, Wi-Fi,
Bluetooth, and 802.15.4.
4.3.1Radio
This subsection describes the fundamental radio technology embedded in the chip that facilitates wireless
communication and data exchange. ESP32-C3 radio consists of the following blocks:
•2.4 GHz receiver
•2.4 GHz transmitter
•Bias and regulators
•Balun and transmit-receive switch
•Clock generator
4.3.1.12.4 GHz Receiver
The 2.4 GHz receiver demodulates the 2.4 GHz RF signal to quadrature baseband signals and converts them
to the digital domain with two high-resolution, high-speed ADCs. To adapt to varying signal channel
conditions, ESP32-C3 integrates RF filters, Automatic Gain Control (AGC), DC offset cancelation circuits, and
baseband filters.
4.3.1.22.4 GHz Transmitter
The 2.4 GHz transmitter modulates the quadrature baseband signals to the 2.4 GHz RF signal, and drives the
antenna with a high-powered CMOS power amplifier. The use of digital calibration further improves the linearity
of the power amplifier.
Additional calibrations are integrated to cancel any radio imperfections, such as:
•Carrier leakage
•I/Q amplitude/phase matching
•Baseband nonlinearities
•RF nonlinearities
•Antenna matching
These built-in calibration routines reduce the cost, time, and specialized equipment required for product
testing.
4.3.1.3Clock Generator
The clock generator produces quadrature clock signals of 2.4 GHz for both the receiver and the transmitter. All
components of the clock generator are integrated into the chip, including inductors, varactors, filters,
regulators and dividers.
Espressif Systems51
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
The clock generator has built-in calibration and self-test circuits. Quadrature clock phases and phase noise
are optimized on chip with patented calibration algorithms which ensure the best performance of the receiver
and the transmitter.
4.3.2Wi-Fi
This subsection describes the chip’s Wi-Fi capabilities, which facilitate wireless communication at a high data
rate.
4.3.2.1Wi-Fi Radio and Baseband
ESP32-C3 Wi-Fi radio and baseband support the following features:
•802.11b/g/n
•802.11n MCS0-7 that supports 20 MHz and 40 MHz bandwidth
•802.11n MCS32
•802.11n 0.4μs guard interval
•Data rate up to 150 Mbps
•RX STBC (single spatial stream)
•Adjustable transmitting power
•Antenna diversity
ESP32-C3 supports antenna diversity with an external RF switch. This switch is controlled by one or
more GPIOs, and used to select the best antenna to minimize the effects of channel imperfections.
4.3.2.2Wi-Fi MAC
ESP32-C3 implements the full 802.11 b/g/n Wi-Fi MAC protocol. It supports the Basic Service Set (BSS) STA
and SoftAP operations under the Distributed Control Function (DCF). Power management is handled
automatically with minimal host interaction to minimize the active duty period.
ESP32-C3 Wi-Fi MAC applies the following low-level protocol functions automatically:
•4 × Virtual Wi-Fi interfaces
•Infrastructure BSS in Station mode, SoftAP mode, Station + SoftAP mode, and promiscuous mode
•RTS protection, CTS protection, Immediate Block ACK
•Fragmentation and defragmentation
•TX/RX A-MPDU, TX/RX A-MSDU
•Transmit opportunity (TXOP)
•Wi-Fi multimedia (WMM)
•GCMP, CCMP, TKIP, WAPI, WEP, BIP, WPA2-PSK/WPA2-Enterprise, and WPA3-PSK/WPA3-Enterprise
•Automatic beacon monitoring (hardware TSF)
•802.11mc FTM
Espressif Systems52
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

4  Functional Description
4.3.2.3Networking Features
Espressif provides libraries for TCP/IP networking, ESP-WIFI-MESH networking, and other networking
protocols over Wi-Fi. TLS 1.0, 1.1 and 1.2 is also supported.
4.3.3Bluetooth LE
This subsection describes the chip’s Bluetooth capabilities, which facilitate wireless communication for
low-power, short-range applications. ESP32-C3 includes a Bluetooth Low Energy subsystem that integrates a
hardware link layer controller, an RF/modem block and a feature-rich software protocol stack. It supports the
core features of Bluetooth 5 and Bluetooth mesh.
4.3.3.1Bluetooth LE PHY
Bluetooth Low Energy radio and PHY in ESP32-C3 support:
•1 Mbps PHY
•2 Mbps PHY for higher data rates
•Coded PHY for longer range (125 Kbps and 500 Kbps)
•HW Listen before talk (LBT)
4.3.3.2Bluetooth LE Link Controller
Bluetooth Low Energy Link Layer Controller in ESP32-C3 supports:
•LE Advertising Extensions, to enhance broadcasting capacity and broadcast more intelligent data
•Multiple advertising sets
•Simultaneous advertising and scanning
•Multiple connections in simultaneous central and peripheral roles
•Adaptive Frequency Hopping (AFH) and channel assessment
•LE Channel Selection Algorithm #2
•Connection parameter update
•High Duty Cycle Non-Connectable Advertising
•LE privacy 1.2
•LE Data Packet Length Extension
•Link Layer Extended Scanner Filter policies
•Low duty cycle directed advertising
•Link layer encryption
•LE Ping
Espressif Systems53
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

5  Electrical Characteristics
5Electrical Characteristics
5.1Absolute Maximum Ratings
Stresses above those listed in Table5-1Absolute Maximum Ratingsmay cause permanent damage to the
device. These are stress ratings only and normal operation of the device at these or any other conditions
beyond those indicated in Section5.2Recommended Operating Conditionsis not implied. Exposure to
absolute-maximum-rated conditions for extended periods may affect device reliability.
Table 5-1. Absolute Maximum Ratings
ParameterDescriptionMinMaxUnit
Input power pins
1
Allowed input voltage0.33.6V
I
output
2
Cumulative IO output current—1000mA
T
ST ORE
Storage temperature40150°C
1
For more information on input power pins, see Section2.5Power Supply.
2
The product proved to be fully functional after all its IO pins were pulled high
while being connected to ground for 24 consecutive hours at ambient tem-
perature of 25 °C.
5.2Recommended Operating Conditions
For recommended ambient temperature, see Section1ESP32-C3 Series Comparison.
Table 5-2. Recommended Operating Conditions
Parameter
1
DescriptionMinTypMaxUnit
VDDA, VDD3P3, VDD3P3_RTCRecommended input voltage3.03.33.6V
VDD3P3_CPU
2, 3
Recommended input voltage3.03.33.6V
VDD_SPI (as input)—3.03.33.6V
I
V DD
Cumulative input current0.5——A
1
See in conjunction with Section2.5Power Supply.
2
If writing to eFuses, the voltage on VDD3P3_CPU should not exceed 3.3 V as the circuits
responsible for burning eFuses are sensitive to higher voltages.
3
If VDD3P3_CPU is used to power VDD_SPI (see Section2.5.2Power Scheme), the voltage
drop on R
SP I
should be accounted for. See also Section5.3VDD_SPI Output Characteristics.
Espressif Systems54
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

5  Electrical Characteristics
5.3VDD_SPI Output Characteristics
Table 5-3. VDD_SPI Internal and Output Characteristics
ParameterDescription
1
TypUnit
R
SP I
VDD_SPI powered by VDD3P3_CPU via R
SP I
for 3.3 V flash_CPU
2
7.5Ω
1
See in conjunction with Section2.5.2Power Scheme.
2
VDD3P3_CPU must be more thanVDD_flash_min + I_flash_max * R
SP I
;
where
•VDD_flash_min– minimum operating voltage of flash_CPU
•I_flash_max– maximum operating current of flash_CPU
5.4DC Characteristics (3.3 V, 25 °C)
Table 5-4. DC Characteristics (3.3 V, 25 °C)
ParameterDescriptionMinTypMaxUnit
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
Low-level input voltage0.3—0.25 × VDD
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
Internal weak pull-up resistor—45—kΩ
R
P D
Internal weak pull-down resistor—45—kΩ
V
IH_nRST
Chip reset release voltage (CHIP_EN voltage
is within the specified range)
0.75 × VDD
1
—VDD
1
+ 0.3V
V
IL_nRST
Chip reset voltage (CHIP_EN voltage is within
the specified range)
0.3—0.25 × VDD
1
V
1
VDD – voltage from a power pin of a respective power domain.
2
V
OH
and V
OL
are measured using high-impedance load.
Espressif Systems55
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

5  Electrical Characteristics
5.5ADC Characteristics
Table 5-5. ADC Characteristics
SymbolParameterMinMaxUnit
DNL (Differential nonlinearity)
1
ADC connected to an external
77LSB
100 nF capacitor; DC signal input;
INL (Integral nonlinearity)
Ambient temperature at 25 °C;
1212LSB
Wi-Fi off
Sampling rate——100
kSPS
2
1
To get better DNL results, you can sample multiple times and apply a filter, or calculate the average
value.
2
kSPS means kilo samples-per-second.
The calibrated ADC results after hardware calibration andsoftwarecalibrationare shown in Table5-6. For
higher accuracy, you may implement your own calibration methods.
Table 5-6. ADC Calibration Results
ParameterDescriptionMinMaxUnit
Total error
ATTEN0, effective measurement range of 0~7501010mV
ATTEN1, effective measurement range of 0~10501010mV
ATTEN2, effective measurement range of 0~13001010mV
ATTEN3, effective measurement range of 0~25003535mV
5.6Current Consumption
5.6.1RF Current Consumption in Active Mode
The current consumption measurements are taken with a 3.3 V supply at 25 °C of ambient temperature at the
RF port. All transmitters’ measurements are based on a 100% duty cycle.
Table 5-7. Wi-Fi Current Consumption Depending on RF Modes
Work Mode
1
DescriptionPeak (mA)
Active (RF working)
TX
802.11b, 1 Mbps, @21 dBm335
802.11g, 54 Mbps, @19 dBm285
802.11n, HT20, MCS7, @18.5 dBm276
802.11n, HT40, MCS7, @18.5 dBm278
RX
802.11b/g/n, HT2084
802.11n, HT4087
Espressif Systems56
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

5  Electrical Characteristics
5.6.2Current Consumption in Other Modes
Table 5-8. Current Consumption in Modem-sleep Mode
Typ
Mode
CPU Frequency
(MHz)
Description
All Peripherals Clocks
Disabled (mA)
All Peripherals Clocks
Enabled (mA)
1
Modem-sleep
2,3
160
CPU is running2328
CPU is idle1621
80
CPU is running1722
CPU is idle1318
1
In practice, the current consumption might be different depending on which peripherals are enabled.
2
In Modem-sleep mode, Wi-Fi is clock gated.
3
In Modem-sleep mode, the consumption might be higher when accessing flash. For a flash rated at
80 Mbit/s, in SPI 2-line mode the consumption is 10 mA.
Table 5-9. Current Consumption in Low-Power Modes
ModeDescriptionTyp (μA)
Light-sleepVDD_SPI and Wi-Fi are powered down, and all GPIOs are high-impedance130
Deep-sleepRTC timer + RTC memory5
Power offCHIP_EN is set to low level, the chip is powered off1
5.7Memory Specifications
The data below is sourced from the memory vendor datasheet. These values are guaranteed through design
and/or characterization but are not fully tested in production. Devices are shipped with the memory
erased.
Table 5-10. Flash Specifications
ParameterDescriptionMinTypMaxUnit
VCC
Power supply voltage (1.8 V)1.651.802.00V
Power supply voltage (3.3 V)2.73.33.6V
F
C
Maximum clock frequency80——MHz
—Program/erase cycles100,000——cycles
T
RET
Data retention time20——years
T
P P
Page program time—0.85ms
T
SE
Sector erase time (4 KB)—70500ms
T
BE1
Block erase time (32 KB)—0.22s
T
BE2
Block erase time (64 KB)—0.33s
T
CE
Chip erase time (16 Mb)—720s
Chip erase time (32 Mb)—2060s
Chip erase time (64 Mb)—25100s
Chip erase time (128 Mb)—60200s
Cont’d on next page
Espressif Systems57
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

5  Electrical Characteristics
Table 5-10 – cont’d from previous page
ParameterDescriptionMinTypMaxUnit
Chip erase time (256 Mb)—70300s
5.8Reliability
Table 5-11. Reliability Qualifications
Test ItemTest ConditionsTest Standard
HTOL (High Temperature
Operating Life)
125 °C, 1000 hoursJESD22-A108
ESD (Electro-Static
Discharge Sensitivity)
HBM (Human Body Mode)
1
± 2000 VJS-001
CDM (Charge Device Mode)
2
± 1000 VJS-002
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
65 °C / 150 °C, 500 cyclesJESD22-A104
uHAST (Highly
Accelerated Stress Test,
unbiased)
130 °C, 85% RH, 96 hoursJESD22-A118
HTSL (High Temperature
Storage Life)
150 °C, 1000 hoursJESD22-A103
LTSL (Low Temperature
Storage Life)
40 °C, 1000 hoursJESD22-A119
1
JEDEC document JEP155 states that 500 V HBM allows safe manufacturing with a standard ESD control process.
2
JEDEC document JEP157 states that 250 V CDM allows safe manufacturing with a standard ESD control process.
Espressif Systems58
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

6  RF Characteristics
6RF Characteristics
This section contains tables with RF characteristics of the Espressif product.
The RF data is measured at the antenna port, where RF cable is connected, including the front-end loss. The
front-end circuit is a 0Ωresistor.
Devices should operate in the center frequency range allocated by regional regulatory authorities. The target
center frequency range and the target transmit power are configurable by software. SeeESPRFTestTooland
TestGuidefor instructions.
Unless otherwise stated, the RF tests are conducted with a 3.3 V (±5%) supply at 25 ºC ambient temperature.
6.1Wi-Fi Radio
Table 6-1. Wi-Fi Frequency
MinTypMax
Parameter(MHz)(MHz)(MHz)
Center frequency of operating channel2412—2484
6.1.1Wi-Fi RF Transmitter (TX) Characteristics
Table 6-2. TX Power with Spectral Mask and EVM Meeting 802.11 Standards
MinTypMax
Rate(dBm)(dBm)(dBm)
802.11b, 1 Mbps—21.0—
802.11b, 11 Mbps—21.0—
802.11g, 6 Mbps—21.0—
802.11g, 54 Mbps—19.0—
802.11n, HT20, MCS0—20.0—
802.11n, HT20, MCS7—18.5—
802.11n, HT40, MCS0—20.0—
802.11n, HT40, MCS7—18.5—
Table 6-3. TX EVM Test
MinTypSL
1
Rate(dB)(dB)(dB)
802.11b, 1 Mbps, @21 dBm—24.510
802.11b, 11 Mbps, @21 dBm—25.010
802.11g, 6 Mbps, @21 dBm—23.05
802.11g, 54 Mbps, @19 dBm—27.525
802.11n, HT20, MCS0, @20 dBm—22.55
Cont’d on next page
Espressif Systems59
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

6  RF Characteristics
Table 6-3 – cont’d from previous page
MinTypSL
1
Rate(dB)(dB)(dB)
802.11n, HT20, MCS7, @18.5 dBm—29.027
802.11n, HT40, MCS0, @20 dBm—22.55
802.11n, HT40, MCS7, @18.5 dBm—28.027
1
SL stands for standard limit value.
6.1.2Wi-Fi RF Receiver (RX) Characteristics
Table 6-4. RX Sensitivity
MinTypMax
Rate(dBm)(dBm)(dBm)
802.11b, 1 Mbps—98.4—
802.11b, 2 Mbps—96.0—
802.11b, 5.5 Mbps—93.0—
802.11b, 11 Mbps—88.6—
802.11g, 6 Mbps—93.8—
802.11g, 9 Mbps—92.2—
802.11g, 12 Mbps—91.0—
802.11g, 18 Mbps—88.4—
802.11g, 24 Mbps—85.8—
802.11g, 36 Mbps—82.0—
802.11g, 48 Mbps—78.0—
802.11g, 54 Mbps—
76.6
—
802.11n, HT20, MCS0—93.6—
802.11n, HT20, MCS1—90.8—
802.11n, HT20, MCS2—88.4—
802.11n, HT20, MCS3—85.0—
802.11n, HT20, MCS4—81.8—
802.11n, HT20, MCS5—77.8—
802.11n, HT20, MCS6—76.0—
802.11n, HT20, MCS7—74.8—
802.11n, HT40, MCS0—90.0—
802.11n, HT40, MCS1—88.0—
802.11n, HT40, MCS2—85.2—
802.11n, HT40, MCS3—82.0—
802.11n, HT40, MCS4—78.8—
802.11n, HT40, MCS5—74.6—
802.11n, HT40, MCS6—73.0—
802.11n, HT40, MCS7—71.4—
Espressif Systems60
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

6  RF Characteristics
Table 6-5. Maximum RX Level
MinTypMax
Rate(dBm)(dBm)(dBm)
802.11b, 1 Mbps—5—
802.11b, 11 Mbps—5—
802.11g, 6 Mbps—5—
802.11g, 54 Mbps—0—
802.11n, HT20, MCS0—5—
802.11n, HT20, MCS7—0—
802.11n, HT40, MCS0—5—
802.11n, HT40, MCS7—0—
Table 6-6. RX Adjacent Channel Rejection
MinTypMax
Rate(dB)(dB)(dB)
802.11b, 1 Mbps—35—
802.11b, 11 Mbps—35—
802.11g, 6 Mbps—31—
802.11g, 54 Mbps—20—
802.11n, HT20, MCS0—31—
802.11n, HT20, MCS7—16—
802.11n, HT40, MCS0—25—
802.11n, HT40, MCS7—11—
6.2Bluetooth 5 (LE) Radio
Table 6-7. Bluetooth LE Frequency
MinTypMax
Parameter
(MHz)(MHz)(MHz)
Center frequency of operating channel2402—2480
6.2.1Bluetooth LE RF Transmitter (TX) Characteristics
Table 6-8. Transmitter Characteristics - Bluetooth LE 1 Mbps
ParameterDescriptionMinTypMaxUnit
RF transmit power
RF power control range24.00020.00dBm
Gain control step—3.00—dB
Carrier frequency offset and drift
Max|f
n
|
n
=0
,
1
,
2
, ..k
—17.00—kHz
Max|f
0−
f
n
|—1.75—kHz
Max|f
n−
f
n−5
|—1.46—kHz
Cont’d on next page
Espressif Systems61
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

6  RF Characteristics
Table 6-8 – cont’d from previous page
ParameterDescriptionMinTypMaxUnit
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
± 2 MHz offset—37.62—dBm
± 3 MHz offset—41.95—dBm
&gt; ± 3 MHz offset—44.48—dBm
Table 6-9. Transmitter Characteristics - Bluetooth LE 2 Mbps
ParameterDescriptionMinTypMaxUnit
RF transmit power
RF power control range24.00020.00dBm
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
± 4 MHz offset—43.55—dBm
± 5 MHz offset—45.26—dBm
&gt; ± 5 MHz offset—45.26—dBm
Table 6-10. Transmitter Characteristics - Bluetooth LE 125 Kbps
ParameterDescriptionMinTypMaxUnit
RF transmit power
RF power control range24.00020.00dBm
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
|
f
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
Min∆f1
max
(for at least
99.9% of all∆f2
max
)
—235.00—kHz
In-band spurious emissions
± 2 MHz offset—37.90—dBm
± 3 MHz offset—41.00—dBm
&gt; ± 3 MHz offset—42.50—dBm
Espressif Systems62
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

6  RF Characteristics
Table 6-11. Transmitter Characteristics - Bluetooth LE 500 Kbps
ParameterDescriptionMinTypMaxUnit
RF transmit power
RF power control range24.00020.00dBm
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
In-band spurious emissions
± 2 MHz offset—37.90—dBm
± 3 MHz offset—41.30—dBm
&gt; ± 3 MHz offset—42.80—dBm
6.2.2Bluetooth LE RF Receiver (RX) Characteristics
Table 6-12. Receiver Characteristics - Bluetooth LE 1 Mbps
ParameterDescriptionMinTypMaxUnit
Sensitivity @30.8% PER——97—dBm
Maximum received signal @30.8% PER——5—dBm
Co-channel C/I——8—dB
Adjacent channel selectivity C/I
F = F0 + 1 MHz—3—dB
F = F0 – 1 MHz—4—dB
F = F0 + 2 MHz—29—dB
F = F0 – 2 MHz—31—dB
F = F0 + 3 MHz—33—dB
F = F0 – 3 MHz—27—dB
F≥F0 + 4 MHz—29—dB
F≤F0 – 4 MHz—38—dB
Image frequency——29—dB
Adjacent channel to image frequency
F = F
image
+ 1 MHz—41—dB
F = F
image
– 1 MHz—33—dB
Out-of-band blocking performance
30 MHz~2000 MHz—5—dBm
2003 MHz~2399 MHz—18—dBm
2484 MHz~2997 MHz—15—dBm
3000 MHz~12.75 GHz—5—dBm
Intermodulation——30—dBm
Espressif Systems63
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

6  RF Characteristics
Table 6-13. Receiver Characteristics - Bluetooth LE 2 Mbps
ParameterDescriptionMinTypMaxUnit
Sensitivity @30.8% PER——93—dBm
Maximum received signal @30.8% PER——3—dBm
Co-channel C/I——10—dB
Adjacent channel selectivity C/I
F = F0 + 2 MHz—7—dB
F = F0 – 2 MHz—7—dB
F = F0 + 4 MHz—28—dB
F = F0 – 4 MHz—26—dB
F = F0 + 6 MHz—26—dB
F = F0 – 6 MHz—27—dB
F≥F0 + 8 MHz—29—dB
F≤F0 – 8 MHz—28—dB
Image frequency——28—dB
Adjacent channel to image frequency
F = F
image
+ 2 MHz—26—dB
F = F
image
– 2 MHz—7—dB
Out-of-band blocking performance
30 MHz~2000 MHz—5—dBm
2003 MHz~2399 MHz—19—dBm
2484 MHz~2997 MHz—16—dBm
3000 MHz~12.75 GHz—5—dBm
Intermodulation——29—dBm
Table 6-14. Receiver Characteristics - Bluetooth LE 125 Kbps
ParameterDescriptionMinTypMaxUnit
Sensitivity @30.8% PER——105—dBm
Maximum received signal @30.8% PER——5—dBm
Co-channel C/I——3—dB
Adjacent channel selectivity C/I
F = F0 + 1 MHz—6—dB
F = F0 – 1 MHz—6—dB
F = F0 + 2 MHz—33—dB
F = F0 – 2 MHz—43—dB
F = F0 + 3 MHz—37—dB
F = F0 – 3 MHz—47—dB
F≥F0 + 4 MHz—40—dB
F≤F0 – 4 MHz—50—dB
Image frequency——40—dB
Adjacent channel to image frequency
F = F
image
+ 1 MHz—50—dB
F = F
image
– 1 MHz—37—dB
Espressif Systems64
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

6  RF Characteristics
Table 6-15. Receiver Characteristics - Bluetooth LE 500 Kbps
ParameterDescriptionMinTypMaxUnit
Sensitivity @30.8% PER——100—dBm
Maximum received signal @30.8% PER——5—dBm
Co-channel C/I——3—dB
Adjacent channel selectivity C/I
F = F0 + 1 MHz—2—dB
F = F0 – 1 MHz—3—dB
F = F0 + 2 MHz—32—dB
F = F0 – 2 MHz—33—dB
F = F0 + 3 MHz—23—dB
F = F0 – 3 MHz—40—dB
F≥F0 + 4 MHz—34—dB
F≤F0 – 4 MHz—44—dB
Image frequency——34—dB
Adjacent channel to image frequency
F = F
image
+ 1 MHz—46—dB
F = F
image
– 1 MHz—23—dB
Espressif Systems65
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

7  Packaging
7Packaging
•For information about tape, reel, and chip marking, please refer to
ESP32-C3 Chip Packaging Information.
•The pins of the chip are numbered in anti-clockwise order starting from Pin 1 in the top view. For pin
numbers and pin names, see also Figure2-1Pin Layout (Top View) for ESP32-C3, ESP32-C3FH4,
ESP32-C3FN4, and ESP32-C3FH8X.
•The recommended land patternsourcefile(dxf)is available for download. You can view the file with
AutodeskViewer.
•For reference PCB layout, please refer toESP32-C3HardwareDesignGuidelines.
Figure 7-1. QFN32 (5×5 mm) Package
Espressif Systems66
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

ESP32-C3 Consolidated Pin Overview
ESP32-C3 Consolidated Pin Overview
Pin
Pin
Pin
Pin Providing
Pin Settings
Analog Function
IO MUX Function
No.
Name
Type
Power
At Reset
After Reset
0
1
0
Type
1
Type
2
Type
1
LNA_IN
Analog
2
VDD3P3
Power
3
VDD3P3
Power
4
XTAL_32K_P
IO
VDD3P3_RTC
XTAL_32K_P
ADC1_CH0
GPIO0
I/O/T
GPIO0
I/O/T
5
XTAL_32K_N
IO
VDD3P3_RTC
XTAL_32K_N
ADC1_CH1
GPIO1
I/O/T
GPIO1
I/O/T
6
GPIO2
IO
VDD3P3_RTC
IE
IE
ADC1_CH2
GPIO2
I/O/T
GPIO2
I/O/T
FSPIQ
I1/O/T
7
CHIP_EN
Analog
8
GPIO3
IO
VDD3P3_RTC
IE
IE
ADC1_CH3
GPIO3
I/O/T
GPIO3
I/O/T
9
MTMS
IO
VDD3P3_RTC
IE
ADC1_CH4
MTMS
I1
GPIO4
I/O/T
FSPIHD
I1/O/T
10
MTDI
IO
VDD3P3_RTC
IE
ADC2_CH0
MTDI
I1
GPIO5
I/O/T
FSPIWP
I1/O/T
11
VDD3P3_RTC
Power
12
MTCK
IO
VDD3P3_CPU
IE
MTCK
I1
GPIO6
I/O/T
FSPICLK
I1/O/T
13
MTDO
IO
VDD3P3_CPU
IE
MTDO
O/T
GPIO7
I/O/T
FSPID
I1/O/T
14
GPIO8
IO
VDD3P3_CPU
IE
IE
GPIO8
I/O/T
GPIO8
I/O/T
15
GPIO9
IO
VDD3P3_CPU
IE, WPU
IE, WPU
GPIO9
I/O/T
GPIO9
I/O/T
16
GPIO10
IO
VDD3P3_CPU
IE
GPIO10
I/O/T
GPIO10
I/O/T
FSPICS0
I1/O/T
17
VDD3P3_CPU
Power
18
VDD_SPI
Power
VDD3P3_CPU
GPIO11
I/O/T
GPIO11
I/O/T
19
SPIHD
IO
VDD_SPI / VDD3P3_CPU
WPU
IE, WPU
SPIHD
I1/O/T
GPIO12
I/O/T
20
SPIWP
IO
VDD_SPI / VDD3P3_CPU
WPU
IE, WPU
SPIWP
I1/O/T
GPIO13
I/O/T
21
SPICS0
IO
VDD_SPI / VDD3P3_CPU
WPU
IE, WPU
SPICS0
O/T
GPIO14
I/O/T
22
SPICLK
IO
VDD_SPI / VDD3P3_CPU
WPU
IE, WPU
SPICLK
O/T
GPIO15
I/O/T
23
SPID
IO
VDD_SPI / VDD3P3_CPU
WPU
IE, WPU
SPID
I1/O/T
GPIO16
I/O/T
24
SPIQ
IO
VDD_SPI / VDD3P3_CPU
WPU
IE, WPU
SPIQ
I1/O/T
GPIO17
I/O/T
25
GPIO18
IO
VDD3P3_CPU
USB_D-
GPIO18
I/O/T
GPIO18
I/O/T
26
GPIO19
IO
VDD3P3_CPU
USB_D+
GPIO19
I/O/T
GPIO19
I/O/T
27
U0RXD
IO
VDD3P3_CPU
IE, WPU
U0RXD
I1
GPIO20
I/O/T
28
U0TXD
IO
VDD3P3_CPU
WPU
U0TXD
O
GPIO21
I/O/T
29
XTAL_N
Analog
30
XTAL_P
Analog
31
VDDA
Power
32
VDDA
Power
33
GND
Power
*
For details, see Section
2
Pins
. Regarding
highlighted
cells, see Section
2.3.3
Restrictions for GPIOs
.
Espressif Systems67
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

ESP32-C3 Chip Series Group Overview
ESP32-C3 Chip Series Group Overview
The ESP32-C3chip series groupis a low-power solution that provides 2.4 GHz Wi-Fi (802.11b/g/n) and
Bluetooth 5.0 connectivity, dedicated to smart home applications. This chip series group consists of the
followingchip series:
•ESP32-C3 series
•ESP8685 series, a cost-down version of ESP32-C3 series
All members within the ESP32-C3 chip series group use a common set of software and reference materials,
including the technical reference manual and hardware design guidelines – SeeRelated Documentation and
Resources.
ESP32-C3ESP8685
Chip revisionv0.4/v1.1v0.4
In-package flashNo/4 MB/8 MB4 MB
Flash extensibilityY—
GPIO count16 or 2215
PackageQFN32 (5×5 mm)QFN28 (4×4 mm)
Espressif Systems68
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Datasheet Versioning
Datasheet Versioning
Datasheet
Version
StatusWatermarkDefinition
v0.1 ~ v0.5
(excluding v0.5)
DraftConfidential
This datasheet is under development for products
in the design stage. Specifications may change
without prior notice.
v0.5 ~ v1.0
(excluding v1.0)
Preliminary
release
Preliminary
This datasheet is actively updated for products in
the verification stage. Specifications may change
before mass production, and the changes will be
documentation in the datasheet’s Revision History.
v1.0 and higherOfficial release—
This datasheet is publicly released for products in
mass production. Specifications are finalized, and
major changes will be communicated viaProduct
ChangeNotifications(PCN).
Any version—
Not
Recommended
for New Design
(NRND)
1
This datasheet is updated less frequently for
products not recommended for new designs.
Any version—
End of Life
(EOL)
2
This datasheet is no longer mtained for products
that have reached end of life.
1
Watermark will be added to the datasheet title page only when all the product variants covered by this
datasheet are not recommended for new designs.
2
Watermark will be added to the datasheet title page only when all the product variants covered by this
datasheet have reached end of life.
Espressif Systems69
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Glossary
Glossary
chip series
A subset of chips within a chip series group with similar core features and specifications2,68
chip series group
A broad group of related chip products that use the same die. For example, ESP32-C3 chip series group
consists of ESP32-C3 chip series and ESP8685 chip series2,68
in-package flash
Flash integrated directly into the chip’s package, and external to the chip die4,35
off-package flash
Flash external to the chip’s package4
peripheral
A hardware component or subsystem within the chip to interface with the outside world16,19
strapping pin
A type of GPIO pin used to configure certain operational settings during the chip’s power-up, and can be
reconfigured as normal GPIO after the chip’s reset30
eFuse parameter
A parameter stored in an electrically programmable fuse (eFuse) memory within a chip. The parameter
can be set by programming EFUSE_PGM_DATA
n_REG registers, and read by reading a register field
named after the parameter30
SPI boot mode
A boot mode in which users load and execute the existing code from SPI flash31
joint download boot mode
A boot mode in which users can download code into flash via the UART or other interfaces (see Table3-3
Chip Boot Mode Control&gt; Note), and load and execute the downloaded code from the flash or SRAM31
eFuse
A one-time programmable (OTP) memory which stores system and user parameters, such as MAC
address, chip revision number, flash encryption key, etc. Value 0 indicates the default state, and value 1
indicates the eFuse has been programmed
35
Espressif Systems70
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Related Documentation and Resources
Related Documentation and Resources
Related Documentation
•ESP32-C3 Technical Reference Manual–Detailed information on how to use the ESP32-C3 memory and periph-
erals.
•ESP32-C3 Hardware Design Guidelines–Guidelines on how to integrate the ESP32-C3 into your hardware prod-
uct.
•ESP32-C3 Series SoC Errata–Descriptions of known errors in ESP32-C3 series of SoCs.
•Certificates
https://espressif.com/en/support/documents/certificates
•ESP32-C3 Product/Process Change Notifications (PCN)
https://espressif.com/en/support/documents/pcns?keys=ESP32-C3
•ESP32-C3 Advisories–Information on security, bugs, compatibility, component reliability.
https://espressif.com/en/support/documents/advisories?keys=ESP32-C3
•Documentation Updates and Update Notification Subscription
https://espressif.com/en/support/download/documents
Developer Zone
•ESP-IDFProgrammingGuideforESP32-C3–Extensive documentation for the ESP-IDF development framework.
•ESP-IDFand other development frameworks on GitHub.
https://github.com/espressif
•ESP32 BBS Forum–Engineer-to-Engineer (E2E) Community for Espressif products where you can post questions,
share knowledge, explore ideas, and help solve problems with fellow engineers.
https://esp32.com/
•ESP-FAQ–A summary document of frequently asked questions released by Espressif.
https://espressif.com/projects/esp-faq/en/latest/index.html
•The ESP Journal–Best Practices, Articles, and Notes from Espressif folks.
https://blog.espressif.com/
•See the tabsSDKs and Demos,Apps,Tools,AT Firmware.
https://espressif.com/en/support/download/sdks-demos
Products
•ESP32-C3 Series SoCs–Browse through all ESP32-C3 SoCs.
https://espressif.com/en/products/socs?id=ESP32-C3
•ESP32-C3 Series Modules–Browse through all ESP32-C3-based modules.
https://espressif.com/en/products/modules?id=ESP32-C3
•ESP32-C3 Series DevKits–Browse through all ESP32-C3-based devkits.
https://espressif.com/en/products/devkits?id=ESP32-C3
•ESP Product Selector–Find an Espressif hardware product suitable for your needs by comparing or applying filters.
https://products.espressif.com/#/product-selector?language=en
Contact Us
•See the tabsSales Questions,Technical Enquiries,Circuit Schematic &amp; PCB Design Review,Get Samples
(Online stores),Become Our Supplier,Comments &amp; Suggestions.
https://espressif.com/en/contact-us/sales-questions
Espressif Systems71
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Revision History
Revision History
DateVersionRelease notes
2026-05-06v2.4
Removed sample marking for ESP32-C3FH8X, as this variant is now in mass
production
2025-12-24v2.3
•Added chip variant ESP32-C3FH8X, see Chapter1ESP32-C3SeriesCom-
parison
•Updated ”Ordering Code” to ”Part Number” in Table1-1ESP32-C3 Series
Comparison
•Added descriptions for the following modules in Chapter4Functional
Description:
–XTAL32K Watchdog Timers (XTWDT)
–World Controller (WCL)
–Clock Glitch Detection
2025-09-04v2.2
•Added Section1.3Chip Revision
•2.3.4Peripheral Pin Assignment: Updated descriptions; added a note
about USB pin swapping to the table
•Updated Figure3-1Visualization of Timing Parameters for the Strapping
Pins
•Added Section5.7Memory Specifications
•Added AppendixDatasheet Versioning
•Other minor updates
2025-04-14v2.1
•Updated CPU CoreMark
®
scode in SectionProduct Overview
•According to updates inCompatibilityAdvisoryforESP32-C3Chip
Revisionv1.1, updated SRAM space in note 5 for TableESP32-C3 Series
Comparison
2024-11-14v2.0
•Added Section2.3.4Peripheral Pin Assignment
•AddedESP32-C3 Chip Series Group Overview
•AddedGlossary
2024-09-11v1.9
Updated pin layout and the number of GPIOs for ESP32-C3FH4X according to
PCN20240702UpgradeofESP32-C3FH4XProduct
Cont’d on next page
Espressif Systems72
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Revision History
Cont’d from previous page
DateVersionRelease notes
2024-07-29v1.8
•Removed  the  ESP32-C3FH4XAZ  variant and  addedCompatibility
AdvisoryforESP32-C3ChipRevisionv1.1in Chapter1ESP32-C3 Series
Comparison
•Updated the default driving strength for each pin in Table2-1PinOverview
&gt; Note 4
•Added flash erase cycles, retention time, maximum clock frequency in
Section4.1.2.1Internal Memory
•Improved the formatting, structure, and wording in the following sections:
–Section2Pins
–Section3Boot Configurations(used to be named as ”Strapping
Pins”)
–Section4Functional Description
•Other minor updates
2024-04-01v1.7
•Marked the ESP32-C3FN4 variant as end of life
•Marked the ESP32-C3FH4AZ variant asNRND
•Marked the ESP32-C3FH4X variant as recommended
2024-01-19v1.6
•Added the new ESP32-C3FH4X and ESP32-C3FH4XAZ variants in Chap-
ter
1ESP32-C3 Series Comparison
•Corrected the PWM duty resolution to 14 bits in Section4.2.1.7LED PWM
Controller
2023-08-11v1.5
•Marked ESP32-C3FN4 asNRND
•Improved the content in the following sections:
–SectionProduct Overview
–Section2Pins
–Section4.1.3.6Power Management Unit
–Section4.2.1.2SPI Controller
–Section5.1Absolute Maximum Ratings
–Section5.2Recommended Operating Conditions
–Section5.3VDD_SPI Output Characteristics
–Section5.5ADC Characteristics
•AddedAppendix A
•Updated the maximum value of ”RF power control range” to 20 dBm in
Section
6.2Bluetooth 5 (LE) Radio
•Other minor updates
Cont’d on next page
Espressif Systems73
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Revision History
Cont’d from previous page
DateVersionRelease notes
2022-12-15v1.4
•Deleted feature ”Antenna diversity” from Section4.3.3.1BluetoothLEPHY
•Deleted feature ”Supports external power amplifier”
•Updated the glitch type of GPIO18 to high-level glitch in TablePower-Up
Glitches on Pins
2022-11-15v1.3
•Updated notes for TablePower-Up Glitches on Pins
•Added links to the Technical Reference Manual and Peripheral Pin Con-
figurations in Chapter4Functional Description
•Added a note about ADC2 error in Section4.2.2.1SAR ADC
•Updated Section4.1.3.8Watchdog Timers
•Added TableADC Calibration Results
•Updated Section5.6.2Current Consumption in Other Modes
•Updated RF transmit power in Section6.2Bluetooth 5 (LE) Radio
•Updated the typo in Section7Packaging
•Updated ChapterRelated Documentation and Resources
2022-04-13v1.2
•Added a new chip variant ESP32-C3FH4AZ;
•Updated FigureESP32-C3 Functional Block Diagram;
•Added the wake up source for Deep-sleep mode in Section4.1.3.6Power
Management Unit
.
2021-10-26v1.1
•Updated FigureESP32-C3 Functional Block Diagramto show power
modes;
•Added CoreMark score in Features;
•Updated Table Pin Description to show default pin functions;
•Updated FigureESP32-C3 Power Schemeand related descriptions;
•Added Table SPI Signals;
•Added note 3 to TableRecommended Operating Conditions;
•Other updates to wording.
2021-05-28v1.0
•Updated power modes;
•Updated Section3Boot Configurations;
•Updated some clock names and their frequencies in Section4.1.3.3
Clock;
•Added clarification about ADC1 and ADC2 in Section4.2.2.1SAR ADC;
•Updated the default configuration of U0RXD and U0TXD after reset in
Table IO MUX;
•Updated sampling rate in TableADC Characteristics;
•Updated TableReliability Qualifications;
•Added the link to recommended PCB land pattern in Chapter7Packaging.
Cont’d on next page
Espressif Systems74
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Revision History
Cont’d from previous page
DateVersionRelease notes
2021-04-23v0.8UpdatedWi-Fi RadioandBluetooth 5 (LE) Radiodata.
2021-04-07v0.7
•Updated information aboutUSB Serial/JTAG Controller;
•Added GPIO2 to Section3Boot Configurations;
•Updated FigureAddress Mapping Structure;
•Added Table IO MUX and TablePower-Up Glitches on Pinsin Section
4.1.3.1IO MUX and GPIO Matrix;
•Updated information about SPI2 in Section4.2.1.2SPI Controller;
•Updated fixed-priority channel scheme in Section4.1.1.2GDMAController;
•Updated TableReliability Qualifications.
2021-01-18v0.6
•Clarified that of the 400 KB SRAM, 16 KB is configured as cache;
•Updated maximum value to standard limit value in TableTX EVM Testin
Section6.1.1Wi-Fi RF Transmitter (TX) Characteristics.
2021-01-13v0.5
•Updated information about Wi-Fi;
•Added connection between in-package flash ports and chip pins to table
notes in Section Pin Definitions;
•Updated FigureESP32-C3 Power Scheme, added FigureVisualization of
Timing Parameters for Power-up and Reset
and TableDescription of Tim-
ing Parameters for Power-up and Resetin Section2.5.2Power Scheme;
•Added FigureVisualization of Timing Parameters for the Strapping Pins
and TableDescription of Timing Parameters for the Strapping Pinsin Sec-
tion3Boot Configurations;
•Updated Table Peripheral Pin Configurations;
•Added Chapter5Electrical Characteristics;
•Added Chapter7Packaging.
2020-11-27v0.4Preliminary version.
Espressif Systems75
Submit Documentation Feedback
ESP32-C3 Series Datasheet v2.4

Disclaimer and Copyright Notice
Information in this document, including URL references, is subject to change without notice.
ALL THIRD PARTY’S INFORMATION IN THIS DOCUMENT IS PROVIDED AS IS WITH NO WARRANTIES TO ITS AUTHENTICITY AND
ACCURACY.
NO WARRANTY IS PROVIDED TO THIS DOCUMENT FOR ITS MERCHANTABILITY, NON-INFRINGEMENT, FITNESS FOR ANY PARTICULAR
PURPOSE, NOR DOES ANY WARRANTY OTHERWISE ARISING OUT OF ANY PROPOSAL, SPECIFICATION OR SAMPLE.
All liability, including liability for infringement of any proprietary rights, relating to use of information in this document is disclaimed. No
licenses express or implied, by estoppel or otherwise, to any intellectual property rights are granted herein.
The Wi-Fi Alliance Member logo is a trademark of the Wi-Fi Alliance. The Bluetooth logo is a registered trademark of Bluetooth SIG.
All trade names, trademarks and registered trademarks mentioned in this document are property of their respective owners, and are
hereby acknowledged.
Copyright © 2026 Espressif Systems (Shanghai) Co., Ltd. All rights reserved.
www.espressif.com