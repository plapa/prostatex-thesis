from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.base import Base
from src.db.tables import * 

engine = create_engine('sqlite:///prostatex.db')
connection = engine.connect()

#Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
s = Session()

if __name__ == "__main__":
    for a in s.query(Patient.ProxID, Image.imagetype).join(Image).filter(Image.registered==1).all():
        print(a)