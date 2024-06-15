import kfp
from kfp import dsl
from kfp.components import create_component_from_func

def train_model_op():
    return dsl.ContainerOp(
        name='Train Model',
        image='hyunjeongshin/final-demo:latest',
        command=['python', 'final_test.py'],
        file_outputs={'model': '/app/model.pkl'}
    )

def apply_tapestry_op(model):
    return dsl.ContainerOp(
        name='Apply Tapestry',
        image='hyunjeongshin/tapestry-smartthings:latest',
        command=['python', 'apply_tapestry.py'],
        arguments=['--model', model],
        file_outputs={'tagged_data': '/app/tagged_data.pkl'}
    )

def notify_smartthings_op(tagged_data):
    return dsl.ContainerOp(
        name='Notify SmartThings',
        image='hyunjeongshin/tapestry-smartthings:latest',
        command=['python', 'notify_smartthings.py'],
        arguments=['--data', tagged_data]
    )

@dsl.pipeline(
    name='Fire Prediction and SmartThings Integration Pipeline',
    description='A pipeline to train model, apply tapestry and notify SmartThings.'
)
def fire_prediction_pipeline():
    train = train_model_op()
    tapestry = apply_tapestry_op(train.outputs['model'])
    notify = notify_smartthings_op(tapestry.outputs['tagged_data'])

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(fire_prediction_pipeline, 'fire_prediction_pipeline.yaml')
