import azureml.core, sys

from azureml.core import Workspace, Datastore, Dataset, Environment, Run, Dataset, Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
    
def main():
    train_file = r"EdwardFry_Microsoft_issueDataset.csv"
    ws = Workspace.from_config()

    # Default datastore 
    def_data_store = ws.get_default_datastore()  # Loads config.json

    # Get the blob storage associated with the workspace
    def_blob_store = Datastore(ws, "workspaceblobstore")

    # Get file storage associated with the workspace
    def_file_store = Datastore(ws, "workspacefilestore")

    # Set data input and output
    xyz_phishing_dataset = Dataset.File.from_files([(def_blob_store, train_file)])
    output_data1 = OutputFileDatasetConfig(destination = (datastore, 'outputdataset/{run-id}'))
    output_data_dataset = output_data1.register_on_complete(name = 'prepared_output_data')

    # Set compute
    compute_name = "aml-compute"
    vm_size = "STANDARD_NC6"
    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print('Found compute target: ' + compute_name)
    else:
        print('Creating a new compute target...')
        provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,  # STANDARD_NC6 is GPU-enabled
                                                                    min_nodes=0,
                                                                    max_nodes=4)
        # create the compute target
        compute_target = ComputeTarget.create(
            ws, compute_name, provisioning_config)

        # Can poll for a minimum number of nodes and for a specific timeout.
        # If no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion(
            show_output=True, min_node_count=None, timeout_in_minutes=20)

        # For a more detailed view of current cluster status, use the 'status' property
        print(compute_target.status.serialize())



    aml_run_config = RunConfiguration()
    # `compute_target` as defined in "Azure Machine Learning compute" section above
    aml_run_config.target = compute_target

    USE_CURATED_ENV = True
    if USE_CURATED_ENV :
        curated_environment = Environment.get(workspace=ws, name="AzureML-Tutorial")
        aml_run_config.environment = curated_environment
    else:
        aml_run_config.environment.python.user_managed_dependencies = False
    
        # Add some packages relied on by data prep step
        aml_run_config.environment.python.conda_dependencies = CondaDependencies.create(
            conda_packages=['pandas','scikit-learn'], 
            pip_packages=['azureml-sdk', 'azureml-dataprep[fuse,pandas]'], 
            pin_sdk_version=False)


    dataprep_source_dir = "./dataprep_src"
    entry_point = "prepare.py"
    # `my_dataset` as defined above
    ds_input = xyz_phishing_dataset.as_named_input('input1')

    # `output_data1`, `compute_target`, `aml_run_config` as defined above
    data_prep_step = PythonScriptStep(
        script_name=entry_point,
        source_directory=dataprep_source_dir,
        arguments=["--input", ds_input.as_download(), "--output", output_data1],
        compute_target=compute_target,
        runconfig=aml_run_config,
        allow_reuse=True
    )

    train_source_dir = "./train_src"
    train_entry_point = "train.py"

    training_results = OutputFileDatasetConfig(name = "training_results",
        destination = def_blob_store)

    
    train_step = PythonScriptStep(
        script_name=train_entry_point,
        source_directory=train_source_dir,
        arguments=["--prepped_data", output_data1.as_input(), "--training_results", training_results],
        compute_target=compute_target,
        runconfig=aml_run_config,
        allow_reuse=True
    )

    # list of steps to run (`compare_step` definition not shown)
    compare_models = [data_prep_step, train_step, compare_step]



    # Build the pipeline
    pipeline1 = Pipeline(workspace=ws, steps=[compare_models])

    #dataset_consuming_step = PythonScriptStep(
    #    script_name="iris_train.py",
    #    inputs=[iris_tabular_dataset.as_named_input("iris_data")],
    #    compute_target=compute_target,
    #    source_directory=project_folder
    #)

    #run_context = Run.get_context()
    #iris_dataset = run_context.input_datasets['iris_data']
    #dataframe = iris_dataset.to_pandas_dataframe()

    ## Within a PythonScriptStep

    #ws = Run.get_context().experiment.workspace

    #step = PythonScriptStep(name="Hello World",
    #                        script_name="hello_world.py",
    #                        compute_target=aml_compute,
    #                        source_directory=source_directory,
    #                        allow_reuse=False,
    #                        hash_paths=['hello_world.ipynb'])

    # Submit the pipeline to be run
    pipeline_run1 = Experiment(ws, 'Compare_Models_Exp').submit(pipeline1)
    pipeline_run1.wait_for_completion()

if __name__ == "__main__":
    sys.exit(main())