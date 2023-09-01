import Levenshtein

def clean_text(text):
    if text is None:
        return ''
    return ' '.join(text.lower().strip().split())

#  Levenshtein distance
def calculate_char_accuracy(detected_text, truth_text):
    return 100 - Levenshtein.distance(clean_text(detected_text), clean_text(truth_text)) * 100 / len(truth_text)


def calculate_word_accuracy(detected_text, truth_text):
    detected_words = clean_text(detected_text).split()
    truth_words = clean_text(truth_text).split()
    total_words = len(truth_words)
    correct_words = sum(1 for dw in detected_words if dw in truth_words)
    return (correct_words / total_words) * 100


def main():
    detected_text_googlecv = """SARKANSAS\nSAMPLE\nSTATE O\n9 CLASS D\n4d DLN 9999999993 DOB 03/05/1960\nNick Sample\nDRIVER'S LICENSE\n1 SAMPLE\n2 NICK\n8 123 NORTH STREET\nCITY, AR 12345\n4a ISS\n03/05/2018\n15 SEX 16 HGT\nM\n5'-10\"\nGREAT SE\n9a END NONE\n12 RESTR NONE\n5 DD 8888888888 1234\n4b EXP\n03/05/2026 MS60\n18 EYES\nBRO\nRKANSAS\n0"""
    detected_text_tesseract = "4d DL 999 as = Ne allo) 2NICK © , q 12 RESTR oe } lick: 5 DD 8888888888 1234 SZ"
    detected_text_easyocr = '''9 , ARKANSAS DRIVER'S LICENSE CLAss D 4d DLN 999999999 3 DOB 03/05/1960 ] 2 SCKPLE 123 NORTH STREET CITY AR 12345 ISS 4b EXP 03/05/2018 03/05/2026 15 SEX 16 HGT 18 EYES 5'-10" BRO 9a END NONE 12 RESTR NONE Ylck Sorble DD 8888888888 1234 THE'''
    truth_text = """USA ARKANSAS 
DRIVER'S LICENSE
9 CLASS D
4d DLN 999999999
3 DOB 03/05/1960
1 SAMPLE 
2 NICK
GREAT SEAL OF THE STATE OF ARKANSAS
8 123 NORTH STREET
CITY, AR 12345
4a ISS
03/05/2018
4b EXP 
03/05/2026
NS60
15 SEX 16 HGT 18 EYES
M 5'-10'' BRO
9a END NONE
12 RESTR NONE
DD 8888888888 1234
Nick Sample"""

    word_accuracy_tesseract = calculate_word_accuracy(detected_text_tesseract, truth_text)
    char_accuracy_tesseract = calculate_char_accuracy(detected_text_tesseract, truth_text)
    word_accuracy_easyocr = calculate_word_accuracy(detected_text_easyocr, truth_text)
    char_accuracy_easyocr = calculate_char_accuracy(detected_text_easyocr, truth_text)
    word_accuracy_google = calculate_word_accuracy(detected_text_googlecv, truth_text)
    char_accuracy_google = calculate_char_accuracy(detected_text_googlecv, truth_text)

    print("             tesseract  |  easyocr  |  google")
    print(f"words         {word_accuracy_tesseract:.2f}%    |  {word_accuracy_easyocr:.2f}%   |  {word_accuracy_google:.2f}%")
    print(f"chars         {char_accuracy_tesseract:.2f}%    |  {char_accuracy_easyocr:.2f}%   |  {char_accuracy_google:.2f}%")


if __name__ == "__main__":
    main()