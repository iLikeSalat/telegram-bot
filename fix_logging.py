import os

def fix_binance_client():
    with open(os.path.join('src', 'binance_client.py'), 'r', encoding='utf-8') as file:
        content = file.readlines()
    
    new_content = []
    for i, line in enumerate(content):
        new_content.append(line)
        if 'async def place_order(' in line:
            new_content.append('        logger.info(f"Attempting to execute order: {symbol} {side} {order_type}")\n')
            new_content.append('        logger.info(f"Parameters: quantity={quantity}, price={price}, stop_price={stop_price}")\n')
            new_content.append('        try:\n')
        elif 'return await self.order_executor.execute_order(' in line:
            new_content[-1] = '            result = await self.order_executor.execute_order(symbol, side, order_type, quantity, price, stop_price, reduce_only)\n'
            new_content.append('            logger.info(f"Order executed successfully: {result}")\n')
            new_content.append('            return result\n')
            new_content.append('        except Exception as e:\n')
            new_content.append('            logger.error(f"Error executing order: {str(e)}")\n')
            new_content.append('            logger.error(f"Binance API details: {e}")\n')
            new_content.append('            raise\n')
    
    with open(os.path.join('src', 'binance_client.py'), 'w', encoding='utf-8') as file:
        file.writelines(new_content)
    
    print("Added logging to binance_client.py")

def fix_trade_executor():
    with open(os.path.join('src', 'trade_executor.py'), 'r', encoding='utf-8') as file:
        content = file.readlines()
    
    new_content = []
    for i, line in enumerate(content):
        new_content.append(line)
        if 'async def execute_signal(self, signal: TradingSignal)' in line:
            new_content.append('        logger.info(f"Starting signal execution: {signal.symbol} {signal.direction}")\n')
            new_content.append('        logger.info(f"Signal details: entry={signal.entry_min}-{signal.entry_max}, SL={signal.stop_loss}")\n')
            new_content.append('        logger.info(f"Take profits: {signal.take_profit_levels}")\n')
    
    with open(os.path.join('src', 'trade_executor.py'), 'w', encoding='utf-8') as file:
        file.writelines(new_content)
    
    print("Added logging to trade_executor.py")

if __name__ == "__main__":
    fix_binance_client()
    fix_trade_executor()
    print("Logging fixes applied successfully")
