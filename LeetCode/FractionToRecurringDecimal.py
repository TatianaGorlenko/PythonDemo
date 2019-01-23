def fractionToDecimal(numerator, denominator):
    '''
    Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
    If the fractional part is repeating, enclose the repeating part in parentheses.
    '''
    float_str = ''
    my_dict = {}
    i = 0
    num = numerator
    den = denominator
    sign = False
    
    if num * den < 0:
        num = abs(numerator)
        den = abs(denominator)
        sign = True
    
    r = num % den
    
    if r == 0:
        return str(numerator // denominator)
    
    int_part = str(num // den)
    
    while True: 
        if r == 0:
            res = int_part + '.' + float_str
            break
        else:
            float_part = r * 10 // den
            if (r, float_part) in my_dict:
                j = my_dict[(r, float_part)]
                float_str = float_str[:j] + '(' + float_str[j:] + ')'
                res = int_part + '.' + float_str
                break
            else:
                my_dict[(r, float_part)] = i

            float_str += str(float_part)
            r = r * 10 % den
            i += 1
    
    if sign:
        return "-" + res
    else:
        return res

if __name__ == '__main__':

    print('1/7 = ', fractionToDecimal(1, 7))
    print('1/4 = ', fractionToDecimal(1, 4))
    print('-2/1 = ', fractionToDecimal(-2, 1))
    print('9/10 = ', fractionToDecimal(9, 10))
    print('5/3 = ', fractionToDecimal(5, 3))



