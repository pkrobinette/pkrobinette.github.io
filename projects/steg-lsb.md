---
layout: default
# title: Steganography
permalink: /sandbox/projects/steg-lsb # can be anything you want
---

# Steganography: Image Steganography Techniques

## 1. Least Significant Bit (LSB) Method TL;DR

Each color channel represents a bit. A single pixel can represent 3 bits and 3 pixels can represent a byte (8 bits) and a flag of whether the message has ended or not. When reading the encoded message, if the channel value is even, it represents a 0 bit; if the channel value is odd it represents a 1 bit. When the flag color channel is 1, the secret message has ended.

| even | 0 |
| --- | --- |
| odd | 1 |
| msg. cont | 0 |
| msg. end | 1 |

The last bit in each color channel (the least significant bit) is flipped to create an even or an odd number if necessary.

### LSB Encoding Example

Say we wanted to encode the phrase “Yo” into the following 2x3 image in its pixel representation:

| (45, 213, 12) | (84, 90, 123) | (231, 97, 109) |
| --- | --- | --- |
| (24, 90, 230) | (56, 124, 56) | (74, 159, 204) |

The three values in each pixel represent red, green, and blue color channels. Represented in binary, the image would look like the following:

| (00101101, 11010101, 00001100) | (01010100, 01011010, 01111011) | (11100111, 01100001, 01101101) |
| --- | --- | --- |
| (00011000, 01011010, 11100110) | (00111000, 01111100, 00111000) | (01001010, 10011111, 11001100) |

The least significant digit in each binary digit is the rightmost of the 8 digits. If this value is switched, it causes the least significant change to the number. For instance, flipping the least significant digit of 255 (1111 1111) results in the value 254 (1111 1110). This is a change of < 1%. If, however, we flip the left most bit (********************the most significant digit)********************, 255 (1111 1111) would then be 127 (0111 1111), a change of 50%. By flipping the least significant bit, we can embed information while imperceptibly changing the rendering of the image.

For each letter in our secret message, we will use 3 pixels or 9 color channels. The first 8 channels are used to embed the character, and the last is used as a flag to indicate if the message continues. 

The secrete message in binary is shown below:

| Ascii | Decimal | Binary |
| --- | --- | --- |
| “Y” | 89 | 01011001 |
| “o” | 111 | 01101111 |

We will use the binary encoding of the secret message to embed the information in the pixels of the cover image. A single bit of this ascii converted binary value will map to an entire color channel. For instance, “Y” will be embedded in the first three pixels (9 color channels) as shown below. If the message bit is 0, the resulting color channel is even; if the message bit is 1, the resulting color channel is odd. This modification is implemented by flipping the least significant bit.

| “Y” → binary | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 1 | Msg. Flag (0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original Image Val. | 45 | 213 | 12 | 84 | 90 | 123 | 231 | 97 | 109 |
| Channel → binary | 00101101 | 11010101 | 00001100 | 01010100 | 01011010 | 01111011 | 11100111 | 01100001 | 01101101 |
| Binary with encoded msg. | 00101100 | 11010101 | 00001100 | 01010101 | 01011011 | 01111010 | 11100110 | 01100001 | 01101100 |
| Image Val. with msg. | 44 | 213 | 12 | 85 | 91 | 122 | 230 | 97 | 108 |

The altered bits and channels are highlighted in red. The last color channel is used to indicate if the message continues or not, where 0 corresponds to a continuing message ,and a 1 indicates the end of the message.

Similarly, for the “o” of the secret message, the process would look like the following:

| “o” → binary | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | Msg. Flag (1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original Image Val. | 24 | 90 | 230 | 56 | 124 | 56 | 74 | 159 | 204 |
| Channel → binary | 00011000 | 01011010 | 11100110 | 00111000 | 01111100 | 00111000 | 01001010 | 10011111 | 11001100 |
| Binary with encoded msg. | 00011000 | 01011011 | 11100111 | 00111000 | 01111101 | 00111001 | 01001011 | 10011111 | 11001101 |
| Image Val. with msg. | 24 | 91 | 231 | 56 | 125 | 57 | 75 | 159 | 205 |

The original image is on top and the resulting altered image is shown beneath it in the tables below.

| (45, 213, 12) | (84, 90, 123) | (231, 97, 109) |
| --- | --- | --- |
| (24, 90, 230) | (56, 124, 56) | (74, 159, 204) |

| (44, 213, 12) | (85, 91, 122) | (230, 97, 108) |
| --- | --- | --- |
| (24, 91, 231) | (56, 125, 57) | (75, 159, 205) |

### LSB Decoding Example

Decoding is then quite simple. By reading the pixels 3 at a time, we can easily decode secret messages embedded with the LSB technique. If the value is even, this corresponds to a 0 bit in our secret message; if the value is odd, this corresponds to a 1 bit in our secret message.

| 44 | 0 |
| --- | --- |
| 213 | 1 |
| 12 | 0 |
| 85 | 1 |
| 91 | 1 |
| 122 | 0 |
| 230 | 0 |
| 97 | 1 |
| 108 (message flag) | 0 (continue) |
| 24 | 0 |
| 91 | 1 |
| 231 | 1 |
| 56 | 0 |
| 125 | 1 |
| 57 | 1 |
| 75  | 1 |
| 159 | 1 |
| 205 (message flag) | 1 (end) |

The secret message is therefore “0101 1001, 0110 1111”, which corresponds to “Yo” in ASCII.

---

## 2. Discrete Cosine Transform (DCT)

# Resources

- [https://medium.com/@stephanie.werli/image-steganography-with-python-83381475da57](https://medium.com/@stephanie.werli/image-steganography-with-python-83381475da57)