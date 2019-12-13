from src.models.architectures.CRFXmasNet import CRFXmasNet
from src.models.architectures.CRFAlexNet import CRFAlexNet



m = CRFAlexNet().architecture()

print(m.summary())
