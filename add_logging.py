import os
import fileinput
import sys

def add_logging_to_binance_client():
    file_path = os.path.join('src', 'binance_client.py')
    
    # Find the execute_signal method in binance_client.py
    with open(file_path, 'r') as file:
        content = file.read()
        
    if 'logger.info(f"Tentative execution ordre sur Binance:' not in content:
        # Add detailed logging before API calls
        for line in fileinput.input(file_path, inplace=True):
            # Add logging before place_order
            if 'async def place_order(' in line:
                sys.stdout.write(line)
                sys.stdout.write('        logger.info(f"Tentative execution ordre sur Binance: {symbol} {side} {order_type}")\n')
                sys.stdout.write('        logger.info(f"Parametres: quantity={quantity}, price={price}, stop_price={stop_price}, reduce_only={reduce_only}")\n')
                sys.stdout.write('        try:\n')
            # Add error catching
            elif 'return await self.order_executor.execute_order(' in line:
                sys.stdout.write('            result = await self.order_executor.execute_order(symbol, side, order_type, quantity, price, stop_price, reduce_only)\n')
                sys.stdout.write('            logger.info(f"Ordre execute avec succes: {result}")\n')
                sys.stdout.write('            return result\n')
                sys.stdout.write('        except Exception as e:\n')
                sys.stdout.write('            logger.error(f"Erreur lors de execution ordre: {str(e)}")\n')
                sys.stdout.write('            logger.error(f"Details de API Binance: {e}")\n')
                sys.stdout.write('            raise\n')
            else:
                sys.stdout.write(line)
    
    print("Logging added to binance_client.py")

def add_logging_to_trade_executor():
    file_path = os.path.join('src', 'trade_executor.py')
    
    # Add detailed logging to execute_signal method
    for line in fileinput.input(file_path, inplace=True):
        if 'async def execute_signal(self, signal: TradingSignal)' in line:
            sys.stdout.write(line)
            sys.stdout.write('        logger.info(f"Debut de execution du signal: {signal.symbol} {signal.direction}")\n')
            sys.stdout.write('        logger.info(f"Details du signal: entry_min={signal.entry_min}, entry_max={signal.entry_max}, stop_loss={signal.stop_loss}")\n')
            sys.stdout.write('        logger.info(f"Take profits: {signal.take_profit_levels}")\n')
        elif 'position_size, leverage, risk_percentage = await self.risk_manager.calculate_position_size(' in line:
            sys.stdout.write(line)
            sys.stdout.write('            logger.info(f"Taille de position calculee: {position_size}, levier: {leverage}, risque: {risk_percentage}%")\n')
        else:
            sys.stdout.write(line)
    
    print("Logging added to trade_executor.py")

if __name__ == "__main__":
    add_logging_to_binance_client()
    add_logging_to_trade_executor()
    print("Additional logging has been added to the bot.")
