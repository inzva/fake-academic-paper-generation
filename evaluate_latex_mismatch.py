import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--latex_file', type=str, default=None, help='LaTeX file to be evaluated', required=True)
opt = parser.parse_args()

text = open(opt.latex_file, 'rb').read().decode(encoding='utf-8')
# delete latex comments
text = re.sub(r'(?<!\\)%.*?\n', '\n', text)

begin_statements = ['\\begin', '{', '$$', '$']
end_statements = ['\\end', '}', '$$', '$']
math_delimiters = ['$$', '$']

stack = []
error_count = 0
i = 0
while i < len(text):
    if not (i>0 and text[i-1] == '\\'):
        for s_i in range(len(begin_statements)):
            if text[i:].startswith(end_statements[s_i]):
                if len(stack) > 0 and stack[-1] == s_i:
                    stack.pop()
                elif s_i in stack:
                    stack.reverse()
                    stack.remove(s_i)
                    stack.reverse()
                elif end_statements[s_i] in math_delimiters:
                    stack.append(s_i)
                else:
                    error_count += 1
                i += len(end_statements[s_i]) - 1
                break
            elif text[i:].startswith(begin_statements[s_i]):
                stack.append(s_i)
                i += len(begin_statements[s_i]) - 1
                break 
    
    i += 1

error_count += len(stack)
print("Stack:", stack)
print("Error Count (Number of Mismatched \\begin..\\end commands, group delimiters {..} and math mode delimiters $..$ $$..$$ ): %d" % error_count)