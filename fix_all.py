import os

def fix_main_py():
    with open('main.py', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Fix the indentation issue
    content = content.replace('with open(config_path, \'r\') as f:\n            config = json.load(f)', 
                             'with open(config_path, \'r\') as f:\n            config = json.load(f)\n            # Set logging level based on debug mode')
    
    with open('main.py', 'w', encoding='utf-8') as file:
        file.write(content)
    
    print("Fixed main.py")

def fix_binance_client():
    with open(os.path.join('src', 'binance_client.py'), 'r', encoding='latin-1') as file:
        content = file.read()
    
    # Replace the problematic line
    content = content.replace('logger.info(f"Tentative', 'logger.info(f"Attempting')
    content = content.replace('execution ordre sur Binance', 'to execute order on Binance')
    content = content.replace('Parametres', 'Parameters')
    content = content.replace('execute avec succes', 'executed successfully')
    content = content.replace('Erreur lors de execution ordre', 'Error executing order')
    content = content.replace('Details de API Binance', 'Binance API details')
    
    with open(os.path.join('src', 'binance_client.py'), 'w', encoding='utf-8') as file:
        file.write(content)
    
    print("Fixed binance_client.py")

if __name__ == "__main__":
    fix_main_py()
    fix_binance_client()
    print("All fixes applied successfully")
