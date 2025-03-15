import heapq
from collections import deque


# Custom Data Structures
class CustomPriorityQueue:
    def __init__(self):  # Time Complexity : O(1) and Space Complexity : O(n)
        self.queue = []

    def push(self, item):  # Time Complexity : O(logn) and Space Complexity : O(1)
        heapq.heappush(self.queue, item)

    def pop(self):  # Time Complexity : O(logn) and Space Complexity : O(1)
        return heapq.heappop(self.queue)

    def __len__(self):  # Time Complexity : O(1) and Space Complexity : O(1)
        return len(self.queue)

    def peek(self):  # Time Complexity : O(1) and Space Complexity : O(1)
        return self.queue[0] if self.queue else None

    def remove(self, item):  # Time Complexity : O(n) and Space Complexity : O(1)
        try:
            self.queue.remove(item)
            heapq.heapify(self.queue)
        except ValueError:
            pass

    def get_all_items(self):  # Time Complexity : O(n) and Space Complexity : O(n)
        return self.queue[:]

    def update_priority(self, old_item, new_item):  # Time Complexity : O(n) and Space Complexity : O(1)
        try:
            idx = self.queue.index(old_item)
            self.queue[idx] = new_item
            heapq.heapify(self.queue)
        except ValueError:
            pass

    def __str__(self):  # Time Complexity : O(n) and Space Complexity : O(n)
        return f"PriorityQueue({self.queue})"


class CustomStack:
    def __init__(self):  # Time Complexity : O(1) and Space Complexity : O(n)
        self.stack = []

    def push(self, item):  # Time Complexity : O(1) and Space Complexity : O(1)
        self.stack.append(item)

    def pop(self):  # Time Complexity : O(1) and Space Complexity : O(1)
        if self.stack:
            return self.stack.pop()
        else:
            return None

    def __len__(self):  # Time Complexity : O(1) and Space Complexity : O(1)
        return len(self.stack)

    def peek(self):  # Time Complexity : O(1) and Space Complexity : O(1)
        return self.stack[-1] if self.stack else None

    def is_empty(self):  # Time Complexity : O(1) and Space Complexity : O(1)
        return len(self.stack) == 0

    def search(self, item):  # Time Complexity : O(n) and Space Complexity : O(1)
        return item in self.stack

    def clear(self):  # Time Complexity : O(1) and Space Complexity : O(1)
        self.stack.clear()


def __str__(self):  # Time Complexity : O(n) and Space Complexity : O(n)
    return f"Stack({self.stack})"


class CustomGraph:  # Time Complexity : O(1) and Space Complexity : O(n+m)
    def __init__(self):
        self.graph = {}

    def add_edge(self, node1, node2, weight):  # Time Complexity : O(1) and Space Complexity : O(1)
        if node1 not in self.graph:
            self.graph[node1] = []
        if node2 not in self.graph:
            self.graph[node2] = []
        self.graph[node1].append((node2, weight))
        self.graph[node2].append((node1, weight))

    def remove_node(self, node):  # Time Complexity : O(n+m) and Space Complexity : O(n+m)
        if node in self.graph:
            del self.graph[node]
            for key in self.graph:
                self.graph[key] = [(n, w) for n, w in self.graph[key] if n != node]

    def shortest_path(self, start, end):  # Time Complexity : O((n+m)logv) and Space Complexity : O(v)
        queue = [(0, start, [])]
        seen = set()
        min_dist = {start: 0}

        while queue:
            (cost, v1, path) = heapq.heappop(queue)

            if v1 in seen:
                continue

            seen.add(v1)
            path = path + [v1]

            if v1 == end:
                return (cost, path)

            for v2, c in self.graph.get(v1, ()):
                if v2 in seen:
                    continue
                prev = min_dist.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    min_dist[v2] = next
                    heapq.heappush(queue, (next, v2, path))

        return float("inf"), []

    def find_all_paths(self, start, end, path=[]):  # Time Complexity : O(V!) and Space Complexity : O(v)
        path = path + [start]
        if start == end:
            return [path]
        if start not in self.graph:
            return []
        paths = []
        for node, weight in self.graph[start]:
            if node not in path:
                new_paths = self.find_all_paths(node, end, path)
                for p in new_paths:
                    paths.append(p)
        return paths

    def __str__(self):  # Time Complexity : O(n+m) and Space Complexity : O(n+m)
        return f"Graph({self.graph})"


class DemandForecast:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.demand_history = deque(maxlen=window_size)
        self.total_demand = 0
        self.num_entries = 0

    def add_demand(self, demand):  # Time Complexity : O(1) and Space Complexity : O(1)
        if len(self.demand_history) == self.window_size:
            self.total_demand -= self.demand_history[0]
        self.demand_history.append(demand)
        self.total_demand += demand
        self.num_entries = min(self.num_entries + 1, self.window_size)

    # calculates the average of recent demands as a forecast
    def forecast_demand(self):  # Time Complexity : O(1) and Space Complexity : O(1)
        if self.num_entries == 0:
            return 0
        return self.total_demand / self.num_entries
        # to get the moving average forecast

    def calculate_trend(self):  # Time Complexity : O(1) and Space Complexity : O(1)
        if len(self.demand_history) < 2:
            return 0
        return (self.demand_history[-1] - self.demand_history[0]) / (len(self.demand_history) - 1)
        # divided by the time between them

    # calculates the moving average over the stored demand values
    def calculate_moving_average(self):  # Time Complexity : O(1) and Space Complexity : O(1)
        if self.num_entries == 0:
            return 0
        return self.total_demand / self.num_entries

    def calculate_seasonal_index(self):  # Time Complexity : O(window_size) and Space Complexity : O(window_size)
        if len(self.demand_history) < self.window_size:
            return [1] * len(self.demand_history)
        seasonal_index = [0] * self.window_size
        for i in range(self.window_size):
            seasonal_index[i] = self.demand_history[i] / self.calculate_moving_average()
        return seasonal_index

    def forecast_with_trend_and_seasonality(
            self):  # Time Complexity : O(window_size) and Space Complexity : O(window_size)
        trend = self.calculate_trend()
        seasonal_indices = self.calculate_seasonal_index()
        base_forecast = self.forecast_demand()
        forecast = []
        for i in range(self.window_size):
            forecast.append((base_forecast + i * trend) * seasonal_indices[i])
        return forecast


# Existing Classes
class Product:
    def __init__(self, product_id, name, price, stock):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.stock = stock
        self.demand_forecast = DemandForecast()

    def update_stock(self, quantity):
        self.stock += quantity

    def __str__(self):
        return f"Product ID: {self.product_id}, Name: {self.name}, Price: {self.price},Stock: {self.stock}"


class Inventory:
    def __init__(self):
        self.products = {}  # Dictionary to store products by product_id
        self.low_stock_heap = CustomPriorityQueue()  # Custom Priority Queue to track products with low stock
        self.transaction_stack = CustomStack()  # Custom Stack to track recent transactions
        self.product_list = []  # List to store all products

    def add_product(self, product):
        if product.product_id in self.products:
            print(f"Product with ID {product.product_id} already exists.")
        else:
            self.products[product.product_id] = product
            self.product_list.append(product)
            self.low_stock_heap.push((product.stock, product.product_id))
            print(f"Product {product.name} added to inventory.")

    def update_product_stock(self, product_id, quantity):
        if product_id in self.products:
            product = self.products[product_id]
            product.update_stock(quantity)
            self.low_stock_heap.push((product.stock, product.product_id))
            self.transaction_stack.push((product_id, quantity))
            print(f"Stock for product ID {product_id} updated.")
        else:
            print(f"Product with ID {product_id} not found.")

    def get_product_info(self, product_id):
        if product_id in self.products:
            return str(self.products[product_id])
        else:
            return f"Product with ID {product_id} not found."

    def list_inventory(self):
        return [str(product) for product in self.product_list]

    def get_low_stock_products(self):
        # Get a list of products with the lowest stock levels
        low_stock_products = []
        for _ in range(min(len(self.low_stock_heap), 5)):
            stock, product_id = self.low_stock_heap.pop()
            low_stock_products.append(self.products[product_id])
        for product in low_stock_products:
            self.low_stock_heap.push((product.stock, product.product_id))
        return low_stock_products

    def get_recent_transactions(self):
        # Get the most recent transactions
        recent_transactions = []
        for _ in range(min(len(self.transaction_stack), 5)):
            recent_transactions.append(self.transaction_stack.pop())
        for transaction in recent_transactions[::-1]:
            self.transaction_stack.push(transaction)
        return recent_transactions[::-1]

    # demand forecasting
    def record_demand(self, product_id, demand):
        if product_id in self.products:
            product = self.products[product_id]
            product.demand_forecast.add_demand(demand)
            print(f"Demand for product ID {product_id} recorded: {demand}")
        else:
            print(f"Product with ID {product_id} not found.")

    def forecast_demand(self, product_id):
        if product_id in self.products:
            product = self.products[product_id]
            return product.demand_forecast.forecast_demand()
        else:
            return f"Product with ID {product_id} not found."

    # New methods to access demand forecast features
    def calculate_trend(self, product_id):
        if product_id in self.products:
            product = self.products[product_id]
            return product.demand_forecast.calculate_trend()
        else:
            return f"Product with ID {product_id} not found."

    def calculate_moving_average(self, product_id):
        if product_id in self.products:
            product = self.products[product_id]
            return product.demand_forecast.calculate_moving_average()
        else:
            return f"Product with ID {product_id} not found."

    def calculate_exponential_smoothing(self, product_id, alpha=0.1):
        if product_id in self.products:
            product = self.products[product_id]
            return product.demand_forecast.calculate_exponential_smoothing(alpha)
        else:
            return f"Product with ID {product_id} not found."

    def calculate_seasonal_index(self, product_id):
        if product_id in self.products:
            product = self.products[product_id]
            return product.demand_forecast.calculate_seasonal_index()
        else:
            return f"Product with ID {product_id} not found."

    def forecast_with_trend_and_seasonality(self, product_id):
        if product_id in self.products:
            product = self.products[product_id]
            return product.demand_forecast.forecast_with_trend_and_seasonality()
        else:
            return f"Product with ID {product_id} not found."


class Order:
    def __init__(self, order_id, product_id, quantity):
        # Initializes an order with an ID, product ID, and quantity
        self.order_id = order_id
        self.product_id = product_id
        self.quantity = quantity
        # Time Complexity: O(1) - Assigning values to attributes is a constant-time operation.
        # Space Complexity: O(1) - The space used by an instance of Order is constant, as it only stores three attributes.

    def __str__(self):
        # Returns a string representation of the order
        return f"Order ID: {self.order_id}, Product ID: {self.product_id}, Quantity: {self.quantity}"
        # Time Complexity: O(1) - Constructing a string from a fixed number of attributes is a constant-time operation.
        # Space Complexity: O(1) - The space used for the string representation is constant.


class OrderProcessing:
    def __init__(self, inventory):
        # Initializes the order processing system with a reference to the inventory
        self.inventory = inventory
        self.order_queue = deque()  # Queue to manage orders
        # Time Complexity: O(1) - Initializing the class and creating a deque is a constant-time operation.
        # Space Complexity: O(1) - The space used for the reference to the inventory and the deque initialization is constant.

    def add_order(self, order):
        # Adds an order to the queue
        self.order_queue.append(order)
        print(f"Order {order.order_id} added to queue.")
        # Time Complexity: O(1) - Appending an item to a deque is a constant-time operation.
        # Space Complexity: O(1) - The space used for the operation is constant, but the overall space for the deque grows linearly with the number of orders.

    def process_order(self):
        # Processes the first order in the queue
        if self.order_queue:
            order = self.order_queue.popleft()  # Removes the first order from the queue
            # Time Complexity: O(1) - Popping an item from the left of a deque is a constant-time operation.
            if order.product_id in self.inventory.products:
                product = self.inventory.products[order.product_id]
                if product.stock >= order.quantity:
                    product.update_stock(-order.quantity)  # Updates the stock
                    # Time Complexity: O(1) - Assuming update_stock is a constant-time operation.
                    self.inventory.transaction_stack.push(
                        (order.product_id, -order.quantity))  # Records the transaction
                    # Time Complexity: O(1) - Pushing to a stack is a constant-time operation.
                    print(f"Order {order.order_id} processed. Stock for product ID {order.product_id} updated.")
                else:
                    print(
                        f"Order {order.order_id} cannot be processed. Not enough stock for product ID {order.product_id}.")
                # Record demand based on processed order
                self.inventory.record_demand(order.product_id, order.quantity)
                # Time Complexity: O(1) - Assuming record_demand is a constant-time operation.
            else:
                print(f"Order {order.order_id} cannot be processed. Product ID {order.product_id} not found.")
        else:
            print("No orders to process.")
        # Space Complexity: O(1) - The space used during the execution of this method is constant, as it only processes one order at a time.

    def list_orders(self):
        # Returns a list of string representations of all orders in the queue
        return [str(order) for order in self.order_queue]
        # Time Complexity: O(n) - Where n is the number of orders in the queue. This is because it iterates over each order to convert it to a string.
        # Space Complexity: O(n) - The space required to store the list of string representations of the orders.


class Supplier:
    def __init__(self, supplier_id, name, region):
        self.supplier_id = supplier_id
        self.name = name
        self.region = region
        self.products = []  # List to store products supplied

    def add_product(self, product):
        self.products.append(product)

    def __str__(self):
        return f"Supplier ID: {self.supplier_id}, Name: {self.name}, Region: {self.region}, Products: {', '.join([product.name for product in self.products])}"


class SupplierNetwork:
    def __init__(self):
        self.suppliers = {}  # Dictionary to store suppliers by supplier_id
        self.supplier_graph = CustomGraph()  # Custom Graph to model supplier relationships
        self.supplier_tree = {}  # Dictionary to store supplier hierarchy

    def add_supplier(self, supplier):
        self.suppliers[supplier.supplier_id] = supplier
        if supplier.region not in self.supplier_tree:
            self.supplier_tree[supplier.region] = []
        self.supplier_tree[supplier.region].append(supplier)
        print(f"Supplier {supplier.name} added to network.")

    def add_supplier_relationship(self, supplier_id1, supplier_id2, weight):
        if supplier_id1 in self.suppliers and supplier_id2 in self.suppliers:
            self.supplier_graph.add_edge(supplier_id1, supplier_id2, weight)
            print(f"Relationship added between {supplier_id1} and {supplier_id2} with weight {weight}.")
        else:
            print("One or both supplier IDs not found.")

    def list_suppliers(self):
        return [str(supplier) for supplier in self.suppliers.values()]

    def list_supplier_hierarchy(self):
        return self.supplier_tree

    def find_shortest_path(self, supplier_id1, supplier_id2):
        return self.supplier_graph.shortest_path(supplier_id1, supplier_id2)

    def find_all_paths(self, supplier_id1, supplier_id2):
        return self.supplier_graph.find_all_paths(supplier_id1, supplier_id2)

    def remove_supplier(self, supplier_id):
        if supplier_id in self.suppliers:
            del self.suppliers[supplier_id]
            self.supplier_graph.remove_node(supplier_id)
            for region, suppliers in self.supplier_tree.items():
                self.supplier_tree[region] = [s for s in suppliers if s.supplier_id != supplier_id]
            print(f"Supplier {supplier_id} removed from network.")
        else:
            print(f"Supplier ID {supplier_id} not found.")

    def update_supplier_info(self, supplier_id, name=None, region=None):
        if supplier_id in self.suppliers:
            supplier = self.suppliers[supplier_id]
            if name:
                supplier.name = name
            if region:
                old_region = supplier.region
                supplier.region = region
                self.supplier_tree[old_region].remove(supplier)
                if region not in self.supplier_tree:
                    self.supplier_tree[region] = []
                self.supplier_tree[region].append(supplier)
            print(f"Supplier {supplier_id} information updated.")
        else:
            print(f"Supplier ID {supplier_id} not found.")

    def find_suppliers_by_region(self, region):
        if region in self.supplier_tree:
            return [str(supplier) for supplier in self.supplier_tree[region]]
        else:
            return f"No suppliers found in region {region}."

    def find_nearest_supplier(self, supplier_id, region=None):
        if supplier_id not in self.suppliers:
            return f"Supplier ID {supplier_id} not found."

        shortest_path = None
        nearest_supplier = None
        for target_supplier_id in self.suppliers:
            if target_supplier_id == supplier_id:
                continue
            if region and self.suppliers[target_supplier_id].region != region:
                continue
            path = self.supplier_graph.shortest_path(supplier_id, target_supplier_id)
            if path[0] != float("inf") and (shortest_path is None or path[0] < shortest_path[0]):
                shortest_path = path
                nearest_supplier = self.suppliers[target_supplier_id]
        if nearest_supplier:
            return f"Nearest supplier: {nearest_supplier.name} with path {shortest_path[1]} and cost {shortest_path[0]}"
        else:
            return "No nearest supplier found."

    def list_supplier_products(self, supplier_id):
        if supplier_id in self.suppliers:
            supplier = self.suppliers[supplier_id]
            return [str(product) for product in supplier.products]
        else:
            return f"Supplier ID {supplier_id} not found."

    def calculate_supplier_importance(self, supplier_id):
        if supplier_id in self.suppliers:
            connections = len(self.supplier_graph.graph.get(supplier_id, []))
            return f"Supplier ID {supplier_id} importance: {connections} connections"
        else:
            return f"Supplier ID {supplier_id} not found."


class Delivery:
    def __init__(self, delivery_id, order_id, delivery_date, priority=1):
        """
        Initializes a Delivery instance with delivery ID, order ID, delivery date, and priority.

        Time Complexity: O(1)
        """
        self.delivery_id = delivery_id
        self.order_id = order_id
        self.delivery_date = delivery_date
        self.priority = priority

    def __lt__(self, other):
        """
        Compares two deliveries based on priority for sorting in priority queue.

        Time Complexity: O(1)
        """
        return self.priority < other.priority

    def __str__(self):
        """
        Returns a formatted string representation of the delivery.

        Time Complexity: O(1)
        """
        return f"Delivery ID: {self.delivery_id}, Order ID: {self.order_id}, Delivery Date: {self.delivery_date}, Priority: {self.priority}"


class DeliverySchedule:
    def __init__(self):
        """
        Initializes a DeliverySchedule instance with a custom priority queue.

        Time Complexity: O(1)
        """
        self.deliveries = CustomPriorityQueue()  # Custom Priority Queue to manage deliveries

    def schedule_delivery(self, delivery):
        """
        Adds a delivery to the priority queue.

        Time Complexity: O(log n), where n is the number of deliveries in the priority queue.
        """
        self.deliveries.push(delivery)
        print(f"Delivery {delivery.delivery_id} scheduled with priority {delivery.priority}.")

    def process_delivery(self):
        """
        Processes the highest-priority delivery from the queue.

        Time Complexity: O(log n), where n is the number of deliveries in the priority queue.
        """
        if len(self.deliveries) > 0:
            delivery = self.deliveries.pop()
            print(f"Processing delivery: {delivery}")
        else:
            print("No deliveries to process.")

    def list_deliveries(self):
        """
        Lists all deliveries in the queue.

        Time Complexity: O(n), where n is the number of deliveries in the priority queue.
        """
        return [str(delivery) for delivery in self.deliveries.queue]


def main():
    # Create instances
    inventory = Inventory()
    order_processing = OrderProcessing(inventory)
    supplier_network = SupplierNetwork()
    delivery_schedule = DeliverySchedule()

    def print_menu():
        print("\n--- Inventory Management System ---")
        print("1. Add Product")
        print("2. Update Product Stock")
        print("3. Record Demand")
        print("4. Forecast Demand")
        print("5. Get Product Info")
        print("6. List Inventory")
        print("7. Get Low Stock Products")
        print("8. Get Recent Transactions")
        print("9. Calculate Trend")
        print("10. Calculate Moving Average")
        print("11. Calculate Seasonal Index")
        print("12. Forecast with Trend and Seasonality")
        print("13. Add Order")
        print("14. Process Order")
        print("15. List Orders")
        print("16. Schedule Delivery")
        print("17. Process Delivery")
        print("18. List Deliveries")
        print("19. Add Supplier")
        print("20. Add Supplier Relationship")
        print("21. List Suppliers")
        print("22. List Supplier Hierarchy")
        print("23. Find Shortest Path Between Suppliers")
        print("24. Find All Paths Between Suppliers")
        print("25. Remove Supplier")
        print("26. Update Supplier Info")
        print("27. Find Suppliers by Region")
        print("28. Find Nearest Supplier")
        print("29. List Supplier Products")
        print("30. Calculate Supplier Importance")
        print("31. Exit")

    print_menu()

    while True:
        print()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            product_id = input("Enter product ID: ").strip()
            name = input("Enter product name: ").strip()
            price = float(input("Enter product price: ").strip())
            stock = int(input("Enter product stock: ").strip())
            print("================================")
            product = Product(product_id, name, price, stock)
            inventory.add_product(product)

        elif choice == "2":
            product_id = input("Enter product ID: ").strip()
            quantity = int(input("Enter quantity to add: ").strip())
            print("================================")
            inventory.update_product_stock(product_id, quantity)

        elif choice == "3":
            product_id = input("Enter product ID: ").strip()
            demand = int(input("Enter demand: ").strip())
            print("================================")
            inventory.record_demand(product_id, demand)

        elif choice == "4":
            product_id = input("Enter product ID: ").strip()
            forecast = inventory.forecast_demand(product_id)
            print("================================")
            print(f"Forecasted demand for product ID {product_id}: {forecast}")

        elif choice == "5":
            product_id = input("Enter product ID: ").strip()
            print("================================")
            print(inventory.get_product_info(product_id))

        elif choice == "6":
            for product in inventory.list_inventory():
                print(product)

        elif choice == "7":
            low_stock_products = inventory.get_low_stock_products()
            for product in low_stock_products:
                print(product)

        elif choice == "8":
            recent_transactions = inventory.get_recent_transactions()
            for transaction in recent_transactions:
                print(transaction)

        elif choice == "9":
            product_id = input("Enter product ID: ").strip()
            trend = inventory.calculate_trend(product_id)
            print("================================")
            print(f"Trend for product ID {product_id}: {trend}")

        elif choice == "10":
            product_id = input("Enter product ID: ").strip()
            moving_average = inventory.calculate_moving_average(product_id)
            print("================================")
            print(f"Moving average for product ID {product_id}: {moving_average}")

        elif choice == "11":
            product_id = input("Enter product ID: ").strip()
            seasonal_index = inventory.calculate_seasonal_index(product_id)
            print("================================")
            print(f"Seasonal index for product ID {product_id}: {seasonal_index}")

        elif choice == "12":
            product_id = input("Enter product ID: ").strip()
            forecast = inventory.forecast_with_trend_and_seasonality(product_id)
            print("================================")
            print(f"Forecast with trend and seasonality for product ID {product_id}: {forecast}")

        elif choice == "13":
            order_id = input("Enter order ID: ").strip()
            product_id = input("Enter product ID: ").strip()
            quantity = int(input("Enter quantity: ").strip())
            print("================================")
            order = Order(order_id, product_id, quantity)
            order_processing.add_order(order)

        elif choice == "14":
            order_processing.process_order()

        elif choice == "15":
            for order in order_processing.list_orders():
                print(order)

        elif choice == "16":
            delivery_id = input("Enter delivery ID: ").strip()
            order_id = input("Enter order ID: ").strip()
            delivery_date = input("Enter delivery date: ").strip()
            priority = int(input("Enter priority: ").strip())
            print("================================")
            delivery = Delivery(delivery_id, order_id, delivery_date, priority)
            delivery_schedule.schedule_delivery(delivery)

        elif choice == "17":
            delivery_schedule.process_delivery()

        elif choice == "18":
            for delivery in delivery_schedule.list_deliveries():
                print(delivery)

        elif choice == "19":
            supplier_id = input("Enter supplier ID: ").strip()
            name = input("Enter supplier name: ").strip()
            region = input("Enter supplier region: ").strip()
            print("================================")
            supplier = Supplier(supplier_id, name, region)
            supplier_network.add_supplier(supplier)

        elif choice == "20":
            supplier_id1 = input("Enter first supplier ID: ").strip()
            supplier_id2 = input("Enter second supplier ID: ").strip()
            weight = int(input("Enter relationship weight: ").strip())
            print("================================")
            supplier_network.add_supplier_relationship(supplier_id1, supplier_id2, weight)

        elif choice == "21":
            for supplier in supplier_network.list_suppliers():
                print(supplier)

        elif choice == "22":
            for region, suppliers in supplier_network.list_supplier_hierarchy().items():
                print(f"Region: {region}")
                for supplier in suppliers:
                    print(f"  {supplier}")

        elif choice == "23":
            supplier_id1 = input("Enter first supplier ID: ").strip()
            supplier_id2 = input("Enter second supplier ID: ").strip()
            print("================================")
            cost, path = supplier_network.find_shortest_path(supplier_id1, supplier_id2)
            if cost != float("inf"):
                print(f"Shortest path: {' -> '.join(path)} with cost {cost}")
            else:
                print("No path found.")

        elif choice == "24":
            supplier_id1 = input("Enter first supplier ID: ").strip()
            supplier_id2 = input("Enter second supplier ID: ").strip()
            print("================================")
            all_paths = supplier_network.find_all_paths(supplier_id1, supplier_id2)
            for path in all_paths:
                print(" -> ".join(path))

        elif choice == "25":
            supplier_id = input("Enter supplier ID: ").strip()
            print("================================")
            supplier_network.remove_supplier(supplier_id)

        elif choice == "26":
            supplier_id = input("Enter supplier ID: ").strip()
            name = input("Enter new supplier name (leave blank to keep current): ").strip()
            region = input("Enter new supplier region (leave blank to keep current): ").strip()
            print("================================")
            supplier_network.update_supplier_info(supplier_id, name if name else None, region if region else None)

        elif choice == "27":
            region = input("Enter region: ").strip()
            print("================================")
            suppliers = supplier_network.find_suppliers_by_region(region)
            if isinstance(suppliers, str):
                print(suppliers)
            else:
                for supplier in suppliers:
                    print(supplier)

        elif choice == "28":
            supplier_id = input("Enter supplier ID: ").strip()
            region = input("Enter region (leave blank to search all regions): ").strip()
            print("================================")
            print(supplier_network.find_nearest_supplier(supplier_id, region if region else None))


        elif choice == "29":
            supplier_id = input("Enter supplier ID: ").strip()
            print("================================")
            print(supplier_network.calculate_supplier_importance(supplier_id))

        elif choice == "30":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
