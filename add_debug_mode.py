import os

def add_debug_mode():
    with open(os.path.join('main.py'), 'r', encoding='utf-8') as file:
        content = file.readlines()
    
    new_content = []
    for line in content:
        new_content.append(line)
        if 'with open(config_path, \'r\') as f:' in line:
            new_content.append('        # Set logging level based on debug mode\n')
            new_content.append('        if config.get("debug", False):\n')
            new_content.append('            logger.setLevel(logging.DEBUG)\n')
            new_content.append('            console_handler.setLevel(logging.DEBUG)\n')
            new_content.append('            logger.debug("Debug mode enabled")\n')
    
    with open(os.path.join('main.py'), 'w', encoding='utf-8') as file:
        file.writelines(new_content)
    
    print("Added debug mode handling")

if __name__ == "__main__":
    add_debug_mode()
