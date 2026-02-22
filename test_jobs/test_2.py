from test_1 import say_hello
import utils.dal.dal as dal


print(say_hello("Nishan"))
print(dal.get_db().db.list_collection_names())
