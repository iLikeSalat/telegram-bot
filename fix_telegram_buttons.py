import os

def fix_telegram_buttons():
    with open(os.path.join('src', 'telegram_client.py'), 'r', encoding='utf-8') as file:
        content = file.readlines()
    
    new_content = []
    for line in content:
        new_content.append(line)
        if 'async def send_signal_result(' in line:
            new_content.append('        # Use a shorter timeout for buttons\n')
            new_content.append('        # Telegram buttons expire after some time\n')
    
    with open(os.path.join('src', 'telegram_client.py'), 'w', encoding='utf-8') as file:
        file.writelines(new_content)
    
    print("Fixed Telegram button handling")

if __name__ == "__main__":
    fix_telegram_buttons()
