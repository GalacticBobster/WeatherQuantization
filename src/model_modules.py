from earth2studio.models.px.pangu import Pangu6
from earth2studio.models.px import FCN

# Pangu6
package = Pangu6.load_default_package()
model   = Pangu6.load_model(package)
print("=== Pangu6 ===")
print(dir(model))          # find what attribute wraps the nn.Module
print(type(model))
for name, module in model.named_modules():
    if name:
        print(f"{type(module).__name__:30s}  {name}")

package = FCN.load_default_package()
model   = FCN.load_model(package)
print("=== FCN ===")
print(dir(model))          # find what attribute wraps the nn.Module
print(type(model))
for name, module in model.named_modules():
    if name:
        print(f"{type(module).__name__:30s}  {name}")



from earth2studio.models.px import AIFS

model = AIFS.load_model(AIFS.load_default_package())

print("=== AIFS Architecture ===")
print(f"Type: {type(model)}")
print(f"\nTop-level attributes: {[x for x in dir(model) if not x.startswith('_')]}")

# Try direct named_modules first
print("\n=== Modules ===")
try:
    for name, module in model.named_modules():
        if name:
            print(f"{type(module).__name__:30s}  {name}")
except:
    # may be wrapped
    try:
        for name, module in model.model.named_modules():
            if name:
                print(f"{type(module).__name__:30s}  {name}")
    except Exception as e:
        print(f"Could not access modules: {e}")


