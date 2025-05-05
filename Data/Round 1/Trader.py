from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    def run(self, state: TradingState):
        print("traderData: ", state.traderData)
        result = {} # 这是一个字典，保存你对每个产品的交易订单
        for product in state.order_depths: # 遍历每种产品(如香蕉、椰子等)
            order_depth = state.order_depths[product] 
            # order_depth是一个OrderDepth对象，包含了买卖盘数据
            orders = [] # 这个列表保存你对每个产品的交易订单
            acceptable_price = 10 # 合理买入价格，这里写死了，其实应该计算
            print("Acceptable price: ", str(acceptable_price))
            print("Buy order depth: ", str(len(order_depth.buy_orders))) # buy_orders是一个字典，每个元素是{价格: 数量}
            print("Sell order depth: ", str(len(order_depth.sell_orders))) # sell_orders是一个字典, 每个元素是{价格: 数量}

            # 接下来判断是否有卖盘(别人的ask)可以让你买
            if len(order_depth.sell_orders)!=0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask)<=acceptable_price: # 可以买入
                    orders.append(Order(product, best_ask, best_ask_amount)) # 数量是负的，表示你是买入
            
            # 再看是否可以卖
            if len(order_depth.buy_orders)!=0:
                best_bid, best_bid_amount =list(order_depth.buy_orders.items())[0]
                if int(best_bid)>=acceptable_price:
                    orders.append(Order(product, best_bid, -best_bid_amount)) 
            
            # 最后把这批订单加入返回结果中
            result[product] = orders
        traderData = "SAMPLE"
        conversions = 1
        return result, conversions, traderData