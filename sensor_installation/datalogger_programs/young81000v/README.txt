The options available for the custom serial output string format primarily include:
Option	Measurement	Size (Bytes)
5	UVW reading	21
6	2D Speed	7
7	3D Speed	7
8	Azim. Angle	6
9	Elev. Angle	6
A	Spd. of Sound	7
B	Temperature	8
C	Checksum	3
E	Error code	4
Call the subroutine ChangeFormat(<com port>, <format string>) to change the string format.
Call the subroutine ChangeFrequency(<com port>, <frequency>) to change the measurement frequency.
The function FormatBytes(<format string>) can be used to determine the expected number of bytes from a serial output string corresponding to the given format.
In calling the SerialInRecord command, the NBytes parameter should be the sum total of the options selected from above in the format string.


If scans are done at the same rate as the measurement frequency, then a frequency should ideally be chosen which evenly divides 1000.
Available frequencies are 4 Hz, 5 Hz, 8 Hz, 10 Hz, 16 Hz, 20 Hz, 32 Hz. (16 and 32 are the only which do not evenly divide 1000)