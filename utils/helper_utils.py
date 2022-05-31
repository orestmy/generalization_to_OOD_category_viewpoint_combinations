def get_digits(mystr):
    digits = []
    for c in mystr:
        if c.isdigit():
            digits.append(c)

    return ''.join(digits)
