# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)


class SimpleBaseModel(BaseModel):
    def get_input_spec(*args, **kwargs):
        return None  # type: ignore

    def get_output_names(*args, **kwargs):
        return None  # type: ignore

    @classmethod
    def from_pretrained(cls):
        return cls()


def test_collection_model_demo():
    """Demo on how to use CollectionModel"""

    class Component1(SimpleBaseModel):
        pass

    class Component2(SimpleBaseModel):
        pass

    class Component3(SimpleBaseModel):
        pass

    @CollectionModel.add_component(Component1)
    @CollectionModel.add_component(Component2)
    class DummyCollection(PretrainedCollectionModel):
        pass

    # Second subclass shouldn't interfere with DummyCollection
    @CollectionModel.add_component(Component1)
    @CollectionModel.add_component(Component2)
    @CollectionModel.add_component(Component3)
    class SecondCollection(PretrainedCollectionModel):
        pass

    # Second subclass shouldn't interfere with DummyCollection
    @CollectionModel.add_component(Component1)
    @CollectionModel.reset_components()
    class ThirdCollection(SecondCollection):
        pass

    # Access class vars via component_classes
    assert DummyCollection.component_classes[0] is Component1
    assert DummyCollection.component_classes[1] is Component2
    assert DummyCollection.component_class_names == ["Component1", "Component2"]

    assert len(ThirdCollection.component_class_names) == 1
    assert len(ThirdCollection.component_classes) == 1

    model = DummyCollection.from_pretrained()

    # Access components via model.components
    assert list(model.components.keys()) == ["Component1", "Component2"]
    assert isinstance(model.components["Component1"], Component1)

    # Instantiate with __init__ directly with positional args
    comp1_instance = Component1()
    comp2_instance = Component2()
    model3 = DummyCollection(comp1_instance, comp2_instance)
    assert model3.components["Component1"] is comp1_instance
    assert model3.components["Component2"] is comp2_instance


def test_missing_from_pretrained():
    """
    Raise Attribute error if any component classes misses from_pretrained or
    from_precompiled method
    """

    class BrokenComponent(SimpleBaseModel):
        # Override from_pretrained with a non-callable value.
        from_pretrained = None  # type: ignore

    @CollectionModel.add_component(BrokenComponent)
    class BrokenCollection(PretrainedCollectionModel):
        pass

    error_msg = (
        "Component 'BrokenComponent' does not have a callable from_pretrained method"
    )
    with pytest.raises(AttributeError, match=error_msg):
        BrokenCollection.from_pretrained()
