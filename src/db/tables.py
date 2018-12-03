from sqlalchemy import Column, String, Integer, Boolean, ForeignKey
from sqlalchemy.orm import backref, relationship

from src.db.base import Base


class Patient(Base):
    __tablename__ = "patients"
    
    ProxID = Column(String(20), primary_key = True)
    lesions = relationship("Lesion", backref="patient")
    images = relationship("Image", backref = "patient")
    
class Lesion(Base):
    __tablename__ = "lesions"
    
    lesion_id = Column(Integer, primary_key=True, autoincrement = True)
    patient_id = Column(Integer, ForeignKey('patients.ProxID'))
    fid = Column(Integer)
    zone = Column(String(2))
    
    reg_i = Column(Integer)
    reg_j = Column(Integer)
    reg_k = Column(Integer)
    ijk = Column(String(10))
    clin_sig = Column(Boolean)
    
    
class Image(Base):
    __tablename__ = "images"
    
    image_id = Column(Integer, primary_key=True, autoincrement = True)
    patient_id = Column(Integer, ForeignKey('patients.ProxID'))
    imagetype = Column(String(150))
    world_matrix = Column(String(250))
    voxel_spacing = Column(String(50))
    registered = Column(Boolean)

class ImageLesion(Base):
    __tablename__ = "image_lesion"
        
    il_id = Column(Integer, primary_key=True, autoincrement = True)
    image_id = Column(Integer, ForeignKey('images.image_id'))
    lesion_id = Column(Integer, ForeignKey('lesions.lesion_id'))