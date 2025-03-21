import os

def fix_binance_client():
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(os.path.join('src', 'binance_client.py'), 'r', encoding=encoding) as file:
                    content = file.read()
                break
            except UnicodeDecodeError:
                continue
        
        # Find the place_order method
        place_order_index = content.find('async def place_order(')
        if place_order_index == -1:
            print("Could not find place_order method")
            return
        
        # Find the next line after the method definition
        next_line_index = content.find('\n', place_order_index) + 1
        
        # Add logging code
        logging_code = '        logger.info(f"Attempting to execute order: {symbol} {side} {order_type}")\n'
        logging_code += '        logger.info(f"Parameters: quantity={quantity}, price={price}, stop_price={stop_price}")\n'
        logging_code += '        try:\n'
        
        # Insert the logging code
        content = content[:next_line_index] + logging_code + content[next_line_index:]
        
        # Find the execute_order call
        execute_order_index = content.find('return await self.order_executor.execute_order(')
        if execute_order_index == -1:
            print("Could not find execute_order call")
            return
        
        # Find the end of the line
        line_end_index = content.find('\n', execute_order_index)
        
        # Get the original line
        original_line = content[execute_order_index:line_end_index]
        
        # Create the replacement code
        replacement_code = '            result = await self.order_executor.execute_order' + original_line[len('return await self.order_executor.execute_order'):]
        replacement_code += '\n            logger.info(f"Order executed successfully: {result}")\n'
        replacement_code += '            return result\n'
        replacement_code += '        except Exception as e:\n'
        replacement_code += '            logger.error(f"Error executing order: {str(e)}")\n'
        replacement_code += '            logger.error(f"Binance API details: {e}")\n'
        replacement_code += '            raise\n'
        
        # Replace the original line
        content = content[:execute_order_index] + replacement_code + content[line_end_index+1:]
        
        # Write the modified content back to the file
        with open(os.path.join('src', 'binance_client.py'), 'w', encoding=encoding) as file:
            file.write(content)
        
        print("Successfully added logging to binance_client.py")
    
    except Exception as e:
        print(f"Error fixing binance_client.py: {str(e)}")

if __name__ == "__main__":
    fix_binance_client()
