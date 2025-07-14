import json
from typing import Dict, List
from json import JSONEncoder
import jsonpickle

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int

class TradingState: # 这个类是系统给你传的
    def __init__(self, traderData, timestamp, listings, order_depths, own_trades, market_trades, position, observations):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths # order_depths是一个字典，key是product name, value是OrderDepth对象，里面包含买单字典和卖单字典
        self.own_trades = own_trades # own_trades记录你上一个TradingState的交易, 是一个字典，key是product_name, value是Trade()对象 
        self.market_trades = market_trades # market_trades记录其它玩家上一个TradingState的交易, 是一个字典, key是product_name, value是Trade()对象
        self.position = position # position是一个字典，key是product_name, value是你持有的数量
        self.observations = observations

    def toJSON(self): # 将TradingState序列化为字符串, 方便存储、查看
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
    
class Trade: # 这个类会显示交易的订单，是TradingState中own_trades和market_trades的值
    def __init__(self, symbol, price, quantity, buyer, seller, timestamp):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer # 你是买家的话，会显示"Submission"
        self.seller = seller # 你是卖家的话，会显示"Submission"
        self.timestamp = timestamp
    def __str__(self):
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")"
    def __repr__(self):
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ")"
    
class OrderDepth: # TradingState类中order_depths的值
    def __init__(self):
        self.buy_orders = {} # 字典中每个元素是{价格：数量}
        self.sell_orders = {} # 字典中每个元素是{价格：数量} 数量是负的
        # 所有买单的价格都应该严格低于所有卖单的价格，不然机器人之间自动撮合了

class ConversionObservation:
    def __init__(self, bidPrice, askPrice, transportFees, exportTariff, importTariff, sugarPrice, sunlightIndex);
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex
    # 你可以在Trader.run()中返回一个整数(conversions), 表示是否请求进行产品转换
    # 想要转换，你必须 1. 持有仓位 2. 转换的数量不能超过你当前持有的数量 3. 需要承担转换成本 4. 转换不是强制行为(不转就传0)

class Order: # 这个类是你要提交的订单，Trader,run()返回值中result的值
    def __init__(self, symbol, price, quantity):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"