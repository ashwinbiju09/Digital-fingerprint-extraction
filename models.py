

class ImageMetadata(db.Model):
    id = db.Column(db.Integer, primary_keys=True)
    filename = db.Column(db.String(255), unique=True, nullable=False)
    date_taken = db.Column(db.DateTime, nullable=False)
