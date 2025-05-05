from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
import statistics


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

class Trader: 
    def __init__(self):
        self.kelp_prices = [] 
        self.kelp_vwap = [] 
        self.kelp_mmmid = []

        self.squid_prices = [] 
        self.squid_vwap = []
        self.squid_mmmid = []
        
        # OU process parameters
        self.ou_kappa = 0.5  # mean reversion speed
        self.ou_theta = 0.0  # long-term mean
        self.ou_sigma = 0.1  # volatility
        self.ou_dt = 1.0     # time step
        self.ou_initialized = False

        self.LIMIT = { 
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50, 
            Product.SQUID_INK: 50
        }

    # Returns buy_order_volume, sell_order_volume
    def take_best_orders(self, product: str, fair_value: int, take_width:float, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int, prevent_adverse: bool = False, adverse_volume: int = 0) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if prevent_adverse:
                if best_ask_amount <= adverse_volume and best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity)) 
                        buy_order_volume += quantity
            else:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity)) 
                        buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if prevent_adverse:
                if (best_bid >= fair_value + take_width) and (best_bid_amount <= adverse_volume):
                    quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity

            else:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity

        return buy_order_volume, sell_order_volume
    
    def market_make(self, product: str, orders: List[Order], bid: int, ask: int, position: int, buy_order_volume: int, sell_order_volume: int) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))  # Sell order
    
        
        return buy_order_volume, sell_order_volume
    
    def clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int) -> List[Order]:
        
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        # fair_for_ask = fair_for_bid = fair

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                # clear_quantity = position_after_take
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                # clear_quantity = abs(position_after_take)
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, method = "mid_price", min_vol = 0) -> float:
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        elif method == "mid_price_with_vol_filter":
            if len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) ==0 or len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) ==0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else: 
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
            return mid_price

    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        # mm_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 20])
        # mm_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 20])
        
        best_ask_fair_value = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        best_bid_fair_value = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])
        
        # Take Orders
        buy_order_volume, sell_order_volume = self.take_best_orders(Product.RAINFOREST_RESIN, fair_value, 0.5, orders, order_depth, position, buy_order_volume, sell_order_volume)
        # Clear Position Orders
        buy_order_volume, sell_order_volume = self.clear_position_order(Product.RAINFOREST_RESIN, fair_value, 1, orders, order_depth, position, buy_order_volume, sell_order_volume)
        # Market Make
        buy_order_volume, sell_order_volume = self.market_make(Product.RAINFOREST_RESIN, orders, best_bid_fair_value + 1, best_ask_fair_value - 1, position, buy_order_volume, sell_order_volume)

        return orders
    

    def kelp_orders(self, order_depth: OrderDepth, timespan:int, width: float, starfruit_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    
            
            # Calculate Fair
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15] # 挂单体积过滤，忽略太小的挂单
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15] # 挂单体积过滤，忽略太小的挂单
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask 
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid 
            
            mmmid_price = (mm_ask + mm_bid) / 2    
            self.kelp_prices.append(mmmid_price) 

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid] 
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap}) 
            # self.kelp_mmmid.append(mmmid_price)
            
            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)
        
            # fair_value = sum([x["vwap"]*x['vol'] for x in self.kelp_vwap]) / sum([x['vol'] for x in self.kelp_vwap])=
            # fair_value = sum(self.kelp_prices) / len(self.kelp_prices)
            fair_value = mmmid_price

            # only taking best bid/ask
            buy_order_volume, sell_order_volume = self.take_best_orders(Product.KELP, fair_value, starfruit_take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, True, 20)
            
            # Clear Position Orders
            buy_order_volume, sell_order_volume = self.clear_position_order(Product.KELP, fair_value, 2, orders, order_depth, position, buy_order_volume, sell_order_volume)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            # Market Make
            buy_order_volume, sell_order_volume = self.market_make(Product.KELP, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders

    def estimate_ou_parameters(self, price_series):
        """Estimate OU process parameters using a simplified approach"""
        if len(price_series) < 10:  # Need enough data points
            return self.ou_kappa, self.ou_theta, self.ou_sigma
            
        # Calculate theta (long-term mean) as the average price
        theta = np.mean(price_series)
        
        # Calculate sigma (volatility) as the standard deviation
        sigma = np.std(price_series)
        
        # Calculate kappa (mean reversion speed) using a simplified approach
        # We'll use the correlation between consecutive price changes and the deviation from mean
        if len(price_series) >= 3:
            # Calculate price changes
            price_changes = np.diff(price_series)
            
            # Calculate deviations from mean
            deviations = np.array(price_series[:-1]) - theta
            
            # Calculate correlation between price changes and deviations
            # This gives us an approximation of kappa
            if len(price_changes) > 1 and len(deviations) > 1:
                # Handle the case where we have enough data for correlation
                try:
                    # Reshape arrays to 2D for np.corrcoef
                    price_changes_2d = price_changes.reshape(-1, 1)
                    deviations_2d = deviations.reshape(-1, 1)
                    correlation = np.corrcoef(price_changes_2d.T, deviations_2d.T)[0, 1]
                    # Convert correlation to kappa (with some scaling)
                    kappa = abs(correlation) * 0.5
                except:
                    # Fallback if correlation calculation fails
                    kappa = 0.5
            else:
                kappa = 0.5
        else:
            kappa = 0.5
            
        return kappa, theta, sigma
    
    def predict_ou_price(self, current_price):
        """Predict next price using OU process"""
        exp_term = np.exp(-self.ou_kappa * self.ou_dt)
        expected_price = self.ou_theta + exp_term * (current_price - self.ou_theta)
        return expected_price
    
    def squid_ink_fair_value(self, order_depth: OrderDepth, method = "mid_price", min_vol = 0) -> float:
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return 0
        
        # Calculate basic price metrics
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        
        # Calculate volume metrics
        bid_volume = order_depth.buy_orders[best_bid]
        ask_volume = -1 * order_depth.sell_orders[best_ask]
        total_volume = bid_volume + ask_volume
        
        # Calculate OBV (On-Balance Volume)
        if total_volume != 0:
            obv = (best_bid * bid_volume - best_ask * ask_volume) / (mid_price * total_volume)
        else:
            obv = 0
            
        # Store historical data for lagged calculations
        self.squid_prices.append(mid_price)
        if len(self.squid_prices) > 30:  # Keep last 30 data points
            self.squid_prices.pop(0)
            
        # Calculate volatility (standard deviation of price)
        if len(self.squid_prices) >= 30:
            vol = np.std(self.squid_prices)
        else:
            vol = 0
            
        # Calculate Bollinger Bands
        window = 20
        if len(self.squid_prices) >= window:
            typical_price = mid_price
            bb_mid = np.mean(self.squid_prices[-window:])
            rolling_std = np.std(self.squid_prices[-window:])
            bb_upper = bb_mid + 2 * rolling_std
            bb_lower = bb_mid - 2 * rolling_std
        else:
            bb_mid = mid_price
            bb_upper = mid_price + 2
            bb_lower = mid_price - 2
            
        # Calculate SMA
        if len(self.squid_prices) >= 23:
            sma23 = np.mean(self.squid_prices[-23:])
        else:
            sma23 = mid_price
            
        # Get lagged mid price
        lag_1 = self.squid_prices[-2] if len(self.squid_prices) >= 2 else mid_price
        
        # Apply linear regression coefficients
        # Coefficients: ['lag_1', 'OBV_lagged', 'vol', 'bb_upper', 'bb_lower', 'sma23']
        # [0.97869853, 0.9281459, 1.30772464, -0.45469355, 0.50658004, -0.03111247]
        # Intercept: 0.9337460631918475
        
        regression_fair_value = (
            0.97869853 * lag_1 +
            0.9281459 * obv +  # Using current OBV as approximation for lagged
            1.30772464 * vol +
            -0.45469355 * bb_upper +
            0.50658004 * bb_lower +
            -0.03111247 * sma23 +
            0.9337460631918475  # Intercept
        )
        
        # Estimate OU parameters if we have enough data
        if len(self.squid_prices) >= 10 and not self.ou_initialized:
            self.ou_kappa, self.ou_theta, self.ou_sigma = self.estimate_ou_parameters(self.squid_prices)
            self.ou_initialized = True
        
        # Predict price using OU process
        ou_fair_value = self.predict_ou_price(mid_price)
        
        # Combine both signals - use a weighted average
        # Give more weight to the regression model (70%) and less to OU (30%)
        combined_fair_value = 0.7 * regression_fair_value + 0.3 * ou_fair_value
        
        return combined_fair_value
    
    def squid_ink_signal(self, order_depth: OrderDepth, method = "mid_price", min_vol = 0) -> bool:
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return False
        
        # Calculate basic price metrics
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        
        # Calculate volume metrics
        bid_volume = order_depth.buy_orders[best_bid]
        ask_volume = -1 * order_depth.sell_orders[best_ask]
        total_volume = bid_volume + ask_volume
        
        # Calculate OBV (On-Balance Volume)
        if total_volume != 0:
            obv = (best_bid * bid_volume - best_ask * ask_volume) / (mid_price * total_volume)
        else:
            obv = 0
            
        # Store historical data for lagged calculations
        self.squid_prices.append(mid_price)
        if len(self.squid_prices) > 30:  # Keep last 30 data points
            self.squid_prices.pop(0)
            
        # Calculate volatility (standard deviation of price)
        if len(self.squid_prices) >= 30:
            vol = np.std(self.squid_prices)
        else:
            vol = 0
            
        # Calculate Bollinger Bands
        window = 20
        if len(self.squid_prices) >= window:
            typical_price = mid_price
            bb_mid = np.mean(self.squid_prices[-window:])
            rolling_std = np.std(self.squid_prices[-window:])
            bb_upper = bb_mid + 2 * rolling_std
            bb_lower = bb_mid - 2 * rolling_std
        else:
            bb_mid = mid_price
            bb_upper = mid_price + 2
            bb_lower = mid_price - 2
            
        # Calculate SMA
        if len(self.squid_prices) >= 23:
            sma23 = np.mean(self.squid_prices[-23:])
        else:
            sma23 = mid_price
            
        # Get lagged mid price
        lag_1 = self.squid_prices[-2] if len(self.squid_prices) >= 2 else mid_price
        
        # Apply linear regression coefficients
        # Coefficients: ['lag_1', 'OBV_lagged', 'vol', 'bb_upper', 'bb_lower', 'sma23']
        # [0.97869853, 0.9281459, 1.30772464, -0.45469355, 0.50658004, -0.03111247]
        # Intercept: 0.9337460631918475
        
        prediction = (
            0.97869853 * lag_1 +
            0.9281459 * obv +  # Using current OBV as approximation for lagged
            1.30772464 * vol +
            -0.45469355 * bb_upper +
            0.50658004 * bb_lower +
            -0.03111247 * sma23 +
            0.9337460631918475  # Intercept
        )
        
        # Return signal: True if prediction > mid_price (good time to long), False otherwise
        return prediction > mid_price
    
    def squid_ink_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders
            
        # Get current market prices
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        
        # Get trading signal
        signal = self.squid_ink_signal(order_depth)
        
        # Calculate order volumes
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Take best orders with a standard approach first
        buy_order_volume, sell_order_volume = self.take_best_orders(
            Product.SQUID_INK, 
            fair_value, 
            width,  # Standard width
            orders, 
            order_depth, 
            position, 
            buy_order_volume, 
            sell_order_volume
        )
        
        # Clear position orders with standard approach
        buy_order_volume, sell_order_volume = self.clear_position_order(
            Product.SQUID_INK, 
            fair_value, 
            width,  # Standard width
            orders, 
            order_depth, 
            position, 
            buy_order_volume, 
            sell_order_volume
        )
        
        # Adjust market making based on signal, but more conservatively
        if signal:  # prediction > mid_price (good time to long)
            # Slightly adjust buy price closer to fair value and sell price further away
            buy_price = max(1, int(fair_value - width * 0.9))  # Slightly closer to fair value
            sell_price = int(fair_value + width * 1.1)  # Slightly further from fair value
        else:  # prediction <= mid_price (good time to short)
            # Slightly adjust sell price closer to fair value and buy price further away
            buy_price = max(1, int(fair_value - width * 1.1))  # Slightly further from fair value
            sell_price = int(fair_value + width * 0.9)  # Slightly closer to fair value
        
        # Market make with slightly adjusted spreads based on signal
        buy_order_volume, sell_order_volume = self.market_make(
            Product.SQUID_INK, 
            orders, 
            buy_price, 
            sell_price, 
            position, 
            buy_order_volume, 
            sell_order_volume
        )
        
        # Add additional orders based on signal only if we have a significant position imbalance
        if abs(position) > position_limit * 0.3:  # Only if position is more than 30% of limit
            if signal and position < 0:  # Positive signal and short position
                # Add a buy order to reduce short position
                if len(order_depth.sell_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    if best_ask <= fair_value + width * 0.5:  # Only if price is reasonable
                        quantity = min(-1 * order_depth.sell_orders[best_ask], -position // 2)  # Buy half of short position
                        if quantity > 0:
                            orders.append(Order(Product.SQUID_INK, best_ask, quantity))
            
            elif not signal and position > 0:  # Negative signal and long position
                # Add a sell order to reduce long position
                if len(order_depth.buy_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    if best_bid >= fair_value - width * 0.5:  # Only if price is reasonable
                        quantity = min(order_depth.buy_orders[best_bid], position // 2)  # Sell half of long position
                        if quantity > 0:
                            orders.append(Order(Product.SQUID_INK, best_bid, -quantity))
        
        return orders

    def run(self, state: TradingState):
        result = {}

        rainforest_resin_fair_value = 10000  # Participant should calculate this value
        rainforest_resin_width = 2
        rainforest_resin_position_limit = 50

        kelp_make_width = 3.5
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timemspan = 10
        
        squid_ink_width = 2
        squid_ink_position_limit = 50
        
        print(state.traderData)

        if Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = state.position[Product.RAINFOREST_RESIN] if Product.RAINFOREST_RESIN in state.position else 0 # 当前rainforest_resin持仓
            rainforest_resin_orders = self.rainforest_resin_orders(state.order_depths[Product.RAINFOREST_RESIN], rainforest_resin_fair_value, rainforest_resin_width, rainforest_resin_position, rainforest_resin_position_limit)
            result[Product.RAINFOREST_RESIN] = rainforest_resin_orders

        if Product.KELP in state.order_depths:
            kelp_position = state.position[Product.KELP] if Product.KELP in state.position else 0 
            kelp_orders = self.kelp_orders(state.order_depths[Product.KELP], kelp_timemspan, kelp_make_width, kelp_take_width, kelp_position, kelp_position_limit)
            result[Product.KELP] = kelp_orders
            
        if Product.SQUID_INK in state.order_depths:
            squid_ink_position = state.position[Product.SQUID_INK] if Product.SQUID_INK in state.position else 0
            # Get trading signal
            squid_ink_signal = self.squid_ink_signal(state.order_depths[Product.SQUID_INK])
            # Get fair value
            squid_ink_fair_value = self.squid_ink_fair_value(state.order_depths[Product.SQUID_INK])
            squid_ink_orders = self.squid_ink_orders(state.order_depths[Product.SQUID_INK], squid_ink_fair_value, squid_ink_width, squid_ink_position, squid_ink_position_limit)
            result[Product.SQUID_INK] = squid_ink_orders

        
        traderData = jsonpickle.encode( { 
            "kelp_prices": self.kelp_prices, 
            "kelp_vwap": self.kelp_vwap,
            "squid_prices": self.squid_prices
        })


        conversions = 1
        
        return result, conversions, traderData