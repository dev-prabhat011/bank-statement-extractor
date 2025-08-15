# Clean up the file again
with open('pdf_to_json_bank_statement.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the correct end - just after main_cli call
for i, line in enumerate(lines):
    if line.strip() == 'main_cli(sys.argv[1:])':
        correct_end = i + 1
        break

# Write clean file
with open('pdf_to_json_bank_statement.py', 'w', encoding='utf-8') as f:
    f.writelines(lines[:correct_end])

print(f"File cleaned. Kept {correct_end} lines.")
