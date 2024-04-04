//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include <string>
#include <cstddef>
#include <atomic>
#include <mutex>
#include <functional>


#include "Wrapper.hpp"


#include "DlSystem/DlEnums.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/DlOptional.hpp"
#include "DlSystem/IBufferAttributes.hpp"
#include "DlSystem/UserMemoryMap.hpp"

#include "SNPE/UserBufferList.hpp"
#include "SNPE/ApplicationBufferMap.hpp"
#include "SNPE/RuntimeConfigList.hpp"
#include "DlContainer/IDlContainer.hpp"

#include "SNPE/RuntimeConfigList.hpp"


#include "SNPE/PSNPE.h"

namespace PSNPE{

enum BuildMode {
  SERIAL = 0,
  PARALLEL = 1
};
/**
 * @brief  Input and output transmission mode
 */
enum InputOutputTransmissionMode {
  sync = 0,
  outputAsync = 1,
  inputOutputAsync = 2
};


struct OutputAsyncCallbackParam : public Wrapper<OutputAsyncCallbackParam, Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t> {
private:
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{NoOpDeleter};


  template<typename DataIndexType>
  using DataIndexReference = WrapperDetail::GenericConstMemberReference
    <OutputAsyncCallbackParam, HandleType, DataIndexType, Snpe_PSNPE_OutputAsyncCallbackParam_GetDataIdx>;


  template<typename ExecuteStatusType>
  using ExecuteStatusReference = WrapperDetail::GenericConstMemberReference
    <OutputAsyncCallbackParam, HandleType, ExecuteStatusType,
     CastingGetter<int, bool, Snpe_PSNPE_OutputAsyncCallbackParam_GetExecuteStatus> >;


  static std::string ErrMsgGetter(Snpe_DlVersion_Handle_t handle){
    return Snpe_PSNPE_OutputAsyncCallbackParam_GetErrorMsg(handle);
  }
  template<typename ErrorMsgType>
  using ErrorMsgReference = WrapperDetail::GenericConstMemberReference
    <OutputAsyncCallbackParam, HandleType, ErrorMsgType, ErrMsgGetter>;

  template<typename CallbackIDType>
  using CallbackIDReference = WrapperDetail::GenericConstMemberReference
    <OutputAsyncCallbackParam, HandleType, CallbackIDType, Snpe_PSNPE_OutputAsyncCallbackParam_GetID>;




public:
  OutputAsyncCallbackParam() = delete;
  OutputAsyncCallbackParam(OutputAsyncCallbackParam&& other) noexcept
    : BaseType(std::move(other))
  {  }

  DataIndexReference<size_t> dataIndex{*this};
  ExecuteStatusReference<bool> executeStatus{*this};
  ErrorMsgReference<std::string> errorMsg{*this};

  CallbackIDReference<size_t> callbackID{*this};
};



struct InputOutputInputAsyncCallbackParam : public Wrapper<InputOutputInputAsyncCallbackParam, Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t> {
private:
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{NoOpDeleter};


  static std::vector<std::string> GetInputs(HandleType handle){
    DlSystem::StringList inputs(moveHandle(Snpe_PSNPE_InputOutputInputAsyncCallbackParam_GetInputs(handle)));

    return std::vector<std::string>(inputs.begin(), inputs.end());
  }

  template<typename InputsType>
  using InputsReference = WrapperDetail::GenericConstMemberReference
     <InputOutputInputAsyncCallbackParam, HandleType, InputsType, GetInputs>;


  static DlSystem::StringList GetInputNames(HandleType handle){
    return moveHandle(Snpe_PSNPE_InputOutputInputAsyncCallbackParam_GetInputNames(handle));
  }
  template<typename InputNamesType>
  using InputNamesReference = WrapperDetail::GenericConstMemberReference
    <InputOutputInputAsyncCallbackParam, HandleType, InputNamesType, GetInputNames>;

  template<typename CallbackIDType>
  using CallbackIDReference = WrapperDetail::GenericConstMemberReference
    <InputOutputInputAsyncCallbackParam, HandleType, CallbackIDType, Snpe_PSNPE_InputOutputInputAsyncCallbackParam_GetID>;


public:
  InputOutputInputAsyncCallbackParam() = delete;
  InputOutputInputAsyncCallbackParam(InputOutputInputAsyncCallbackParam&& other) noexcept
  : BaseType(std::move(other))
  {  }

  InputsReference<std::vector<std::string>> inputs{*this};
  InputNamesReference<DlSystem::StringList> inputNames{*this};
  CallbackIDReference<size_t> callbackID{*this};

};





struct InputOutputAsyncCallbackParam : public Wrapper<InputOutputAsyncCallbackParam, Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t, true> {
private:
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{NoOpDeleter};

  template<typename DataIndexType>
  using DataIndexReference = WrapperDetail::GenericConstMemberReference
    <InputOutputAsyncCallbackParam, HandleType, DataIndexType, Snpe_PSNPE_InputOutputAsyncCallbackParam_GetDataIdx>;

  static bool GetExecuteStatus(HandleType handle){
    return Snpe_PSNPE_InputOutputAsyncCallbackParam_GetExecuteStatus(handle);
  }
  template<typename ExecuteStatusType>
  using ExecuteStatusReference = WrapperDetail::GenericConstMemberReference
    <InputOutputAsyncCallbackParam, HandleType, ExecuteStatusType, GetExecuteStatus>;

  static std::string ErrMsgGetter(Snpe_DlVersion_Handle_t handle){
    return Snpe_PSNPE_OutputAsyncCallbackParam_GetErrorMsg(handle);
  }
  template<typename ErrorMsgType>
  using ErrorMsgReference = WrapperDetail::GenericConstMemberReference
    <InputOutputAsyncCallbackParam, HandleType, ErrorMsgType, ErrMsgGetter>;



  // This should work
  static ApplicationBufferMap GetOutputMap(HandleType handle){
    return moveHandle(Snpe_PSNPE_InputOutputAsyncCallbackParam_GetOutputMap_Ref(handle), true);
  }

  template<typename OutputMapType>
  using OutputMapReference = WrapperDetail::GenericConstMemberReference
    <InputOutputAsyncCallbackParam, HandleType, OutputMapType, GetOutputMap>;

  template<typename CallbackIDType>
  using CallbackIDReference = WrapperDetail::GenericConstMemberReference
    <InputOutputAsyncCallbackParam, HandleType, CallbackIDType, Snpe_PSNPE_InputOutputAsyncCallbackParam_GetID>;

public:

  InputOutputAsyncCallbackParam(InputOutputAsyncCallbackParam&& other) noexcept
    : BaseType(std::move(other))
  {  }

  DataIndexReference<size_t> dataIndex{*this};
  OutputMapReference<ApplicationBufferMap> outputMap{*this}; /// OOOH, this will be super tricky to not have a copy every time
  ExecuteStatusReference<bool> executeStatus{*this};
  ErrorMsgReference<std::string> errorMsg{*this};
  CallbackIDReference<size_t> callbackID{*this};
};

/**
 * @brief  This callback is called when the output data is ready, only use for Output Async mode
 */
using OutputAsyncCallbackFunc = std::function<void(OutputAsyncCallbackParam)>;
/**
 * @brief  This callback is called when the output data is ready, only use for Output-Input Async mode
 */
using InputOutputAsyncCallbackFunc = std::function<void(InputOutputAsyncCallbackParam)>;
/**
 * @brief   This callback is called when the input data is ready,only use for Output-Input Async mode
 */
using InputOutputAsyncInputCallback = std::function<std::shared_ptr<ApplicationBufferMap>(InputOutputInputAsyncCallbackParam)>;


struct BuildConfig final {
  BuildMode buildMode = BuildMode::SERIAL; ///< Specify build in serial mode or parallel mode
  zdl::DlContainer::IDlContainer* container;///< The opened container ptr
  zdl::DlSystem::StringList outputBufferNames;///< Specify the output layer name
  zdl::DlSystem::StringList outputTensors;///< Specify the output layer name
  RuntimeConfigList runtimeConfigList;///< The runtime config list for PSNPE, @see RuntimeConfig
  size_t inputThreadNumbers = 1;///< Specify the number of threads used in the execution phase to process input data, only used in inputOutputAsync mode
  size_t outputThreadNumbers = 1;///< Specify the number of threads used in the execution phase to process output data, only used in inputOutputAsync and outputAsync mode
  OutputAsyncCallbackFunc outputCallback;///< The callback to deal with output data ,only used in outputAsync mode
  InputOutputAsyncCallbackFunc inputOutputCallback;///< The callback to deal with output data ,only used in inputOutputAsync mode
  InputOutputAsyncInputCallback inputOutputInputCallback;///< The callback to deal with input data ,only used in inputOutputAsync mode
  InputOutputTransmissionMode inputOutputTransmissionMode = InputOutputTransmissionMode::sync;///< Specify execution mode
  zdl::DlSystem::ProfilingLevel_t profilingLevel = zdl::DlSystem::ProfilingLevel_t::OFF;///< Specify profiling level for Diaglog
  uint64_t encode[2] = {0, 0};
  bool enableInitCache = false;
  std::string platformOptions;
  std::string diaglogOutputDir = "./diaglogs/"; ///< Specify a diaglog output directory to save the generated Diaglog files.

  size_t callbackID{};
};





class PSNPE  : public Wrapper<PSNPE, Snpe_PSNPE_Handle_t> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_PSNPE_Delete};
//  struct BuildConfigInternal : public Wrapper<BuildConfigInternal, Snpe_BuildConfig_Handle_t>{
//
//  };
public:
  PSNPE()
    : BaseType(Snpe_PSNPE_Create())
  {  }

private:

  template<typename WrapperCallbackType>
  static std::unordered_map<size_t, WrapperCallbackType>& getCallbackMap(){
    static std::unordered_map<size_t, WrapperCallbackType> toret;
    return toret;
  }
  template<typename>
  static std::mutex& getCallbackMapMutex(){
    static std::mutex mtx;
    return mtx;
  }

  static void outputCallbackTrampoline(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t paramHandle){
    OutputAsyncCallbackParam param(moveHandle(paramHandle));
    std::function<void(OutputAsyncCallbackParam)> callback;
    {
      std::lock_guard<std::mutex> lk(getCallbackMapMutex<OutputAsyncCallbackFunc>());
      callback = getCallbackMap<OutputAsyncCallbackFunc>()[param.callbackID];
    }
    callback(std::move(param));
  }
  static void inputOutputCallbackTrampoline(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t paramHandle){
    InputOutputAsyncCallbackParam param(moveHandle(paramHandle));
    std::function<void(InputOutputAsyncCallbackParam)> callback;
    {
      std::lock_guard<std::mutex> lk(getCallbackMapMutex<InputOutputAsyncCallbackFunc>());
      callback = getCallbackMap<InputOutputAsyncCallbackFunc>()[param.callbackID];
    }
    callback(std::move(param));
  }

  static Snpe_ApplicationBufferMap_Handle_t inputOutputInputCallbackTrampoline(
    Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t paramHandle
  ){
    InputOutputInputAsyncCallbackParam param(moveHandle(paramHandle));

    std::function<std::shared_ptr<ApplicationBufferMap>(InputOutputInputAsyncCallbackParam)> callback;
    {
      std::lock_guard<std::mutex> lk(getCallbackMapMutex<InputOutputAsyncInputCallback>());
      callback = getCallbackMap<InputOutputAsyncInputCallback>()[param.callbackID];
    }
    auto abm = callback(std::move(param));
    return WrapperDetail::HandleReleaser::release(*abm);
  }

  template<typename WrapperCallbackType, typename CAPICallbackType, CAPICallbackType CapiCallback>
  class CallbackIdManager{
  public:
    ~CallbackIdManager(){
      clear();
    }
    std::pair<size_t,CAPICallbackType> registerCallback(WrapperCallbackType func){
      size_t id = get();

      std::lock_guard<std::mutex> lk(getCallbackMapMutex<WrapperCallbackType>());
      getCallbackMap<WrapperCallbackType>()[id] = std::move(func);
      return {id, CapiCallback};
    }
  private:
    size_t m_CallbackId{};

    void clear(){
      if(m_CallbackId){
        std::lock_guard<std::mutex> lk(getCallbackMapMutex<WrapperCallbackType>());
        getCallbackMap<WrapperCallbackType>().erase(m_CallbackId);
      }
    }

    size_t get(){
      static std::atomic<size_t> id{0};
      clear();
      m_CallbackId = ++id;
      return m_CallbackId;
    }

  };
  CallbackIdManager<OutputAsyncCallbackFunc,
                    void(*)(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t),
                    outputCallbackTrampoline> outputCallbackIdManager;

  CallbackIdManager<InputOutputAsyncCallbackFunc,
                    void(*)(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t),
                    inputOutputCallbackTrampoline> inputOutputCallbackIdManager;

  CallbackIdManager<InputOutputAsyncInputCallback,
                    Snpe_ApplicationBufferMap_Handle_t(*)(Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t),
                    inputOutputInputCallbackTrampoline> inputOutputInputCallbackIdManager;


public:



  bool build(BuildConfig& buildConfig) noexcept{
    // Copy the BuildConfig across the CAPI boundary

    Snpe_BuildConfig_Handle_t bcHandle = Snpe_BuildConfig_Create();

    Snpe_BuildConfig_SetBuildMode(bcHandle, static_cast<Snpe_PSNPE_BuildMode_t>(buildConfig.buildMode));
    Snpe_BuildConfig_SetContainer(bcHandle, getHandle(buildConfig.container));
    Snpe_BuildConfig_SetOutputBufferNames(bcHandle, getHandle(buildConfig.outputBufferNames));
    Snpe_BuildConfig_SetOutputTensors(bcHandle, getHandle(buildConfig.outputTensors));
    Snpe_BuildConfig_SetRuntimeConfigList(bcHandle, getHandle(buildConfig.runtimeConfigList));

    Snpe_BuildConfig_SetInputThreadNumbers(bcHandle, buildConfig.inputThreadNumbers);
    Snpe_BuildConfig_SetOutputThreadNumbers(bcHandle, buildConfig.outputThreadNumbers);


    if(buildConfig.outputCallback){
      auto id_callback = outputCallbackIdManager.registerCallback(buildConfig.outputCallback);
      Snpe_BuildConfig_SetOutputCallbackID(bcHandle, id_callback.first);
      Snpe_BuildConfig_SetOutputCallback(bcHandle, id_callback.second);
    }

    if(buildConfig.inputOutputCallback){
      auto id_callback = inputOutputCallbackIdManager.registerCallback(buildConfig.inputOutputCallback);
      Snpe_BuildConfig_SetInputOutputCallbackID(bcHandle, id_callback.first);
      Snpe_BuildConfig_SetInputOutputCallback(bcHandle, id_callback.second);
    }

    if(buildConfig.inputOutputInputCallback){
      auto id_callback = inputOutputInputCallbackIdManager.registerCallback(buildConfig.inputOutputInputCallback);
      Snpe_BuildConfig_SetInputOutputInputCallbackID(bcHandle, id_callback.first);
      Snpe_BuildConfig_SetInputOutputInputCallback(bcHandle, id_callback.second);
    }


    Snpe_BuildConfig_SetInputOutputTransmissionMode(bcHandle,
      static_cast<Snpe_PSNPE_InputOutputTransmissionMode_t>(buildConfig.inputOutputTransmissionMode));

    Snpe_BuildConfig_SetProfilingLevel(bcHandle, static_cast<Snpe_ProfilingLevel_t>(buildConfig.profilingLevel));
    Snpe_BuildConfig_SetEncode(bcHandle, buildConfig.encode[0], buildConfig.encode[1]);
    Snpe_BuildConfig_SetEnableInitCache(bcHandle, buildConfig.enableInitCache);
    Snpe_BuildConfig_SetPlatformOptions(bcHandle, buildConfig.platformOptions.c_str());
    Snpe_BuildConfig_SetDiaglogOutputDir(bcHandle, buildConfig.diaglogOutputDir.c_str());


    auto status = Snpe_PSNPE_Build(handle(), bcHandle);
    Snpe_BuildConfig_Delete(bcHandle);


    return status == SNPE_SUCCESS;
  }

  /**
   * @brief Execute snpe instances in Async Output mode and Sync mode
   *
   * @param[in] inputBufferList A list of user buffers that contains the input data
   *
   * @param[in,out] outputBufferList A list of user buffers that will hold the output data
   *
   */
  bool execute(UserBufferList& inputBufferList, UserBufferList& outputBufferList) noexcept{
    return  SNPE_SUCCESS == Snpe_PSNPE_Execute(handle(), getHandle(inputBufferList), getHandle(outputBufferList));
  }

  /**
   * @brief  Execute snpe instances in Async Input/Output mode
   *
   * @param[in]inputMap A map of input buffers that contains input data. The names of buffers
   *                     need to be matched with names retrived through getInputTensorNames()
   *
   * @param dataIndex Index of the input data
   *
   * @param isTF8buff Whether prefer to using 8 bit quantized element for inference
   *
   * @return True if executed successfully; flase, otherwise.
   */
  bool executeInputOutputAsync(const DlSystem::StringList& inputMap, size_t dataIndex, bool isTF8buff, bool isTF8Outputbuff) noexcept{
    return SNPE_SUCCESS == Snpe_PSNPE_ExecuteInputOutputAsync(handle(), getHandle(inputMap), dataIndex, isTF8buff, isTF8Outputbuff);
  }
  bool executeInputOutputAsync(const std::vector<std::string>& inputMap, size_t dataIndex, bool isTF8buff, bool isTF8Outputbuff) noexcept{
    DlSystem::StringList sl(inputMap.size());
    for(auto&& e : inputMap) sl.append(e.c_str());
    return executeInputOutputAsync(sl, dataIndex, isTF8buff, isTF8Outputbuff);
  }

  bool executeInputOutputAsync(const DlSystem::StringList& inputMap, size_t dataIndex, bool isTF8buff) noexcept{
    return executeInputOutputAsync(inputMap, dataIndex, isTF8buff, isTF8buff);
  }
  bool executeInputOutputAsync(const std::vector<std::string>& inputMap, size_t dataIndex, bool isTF8buff) noexcept{
    return executeInputOutputAsync(inputMap, dataIndex, isTF8buff, isTF8buff);
  }



  /**
   * @brief Returns the input layer names of the network.
   *
   * @return StringList which contains the input layer names
   */
  const DlSystem::StringList getInputTensorNames() const noexcept{
    return moveHandle(Snpe_PSNPE_GetInputTensorNames(handle()));
  }

  /**
   * @brief Returns the output layer names of the network.
   *
   * @return StringList which contains the output layer names
   */
  const DlSystem::StringList getOutputTensorNames() const noexcept{
    return moveHandle(Snpe_PSNPE_GetOutputTensorNames(handle()));
  }

  /**
   * @brief Returns the input tensor dimensions of the network.
   *
   * @return TensorShape which contains the dimensions.
   */
  const DlSystem::TensorShape getInputDimensions() const noexcept{
    return moveHandle(Snpe_PSNPE_GetInputDimensions(handle()));
  }

  const zdl::DlSystem::TensorShape getInputDimensions(const char *name) const noexcept{
    return moveHandle(Snpe_PSNPE_GetInputDimensions_Name(handle(), name));
  }

  /**
   * @brief Returns attributes of buffers.
   *
   * @see zdl::SNPE
   *
   * @return BufferAttributes of input/output tensor named.
   */
  zdl::DlSystem::TensorShape getBufferAttributesDims(const char *name) const noexcept{
    return moveHandle(Snpe_PSNPE_GetBufferAttributesDims(handle(), name));
  }

  DlSystem::Optional<DlSystem::IBufferAttributes*> getInputOutputBufferAttributes(const char *name) const noexcept{
    return {
      new DlSystem::IBufferAttributes(moveHandle(Snpe_PSNPE_GetInputOutputBufferAttributes(handle(), name))),
      DlSystem::Optional<DlSystem::IBufferAttributes*>::LIFECYCLE::POINTER_OWNED
    };
  }
  /* To be deprecated, please use new api registerMemoryMappedBuffers */
  bool registerIonBuffers(const DlSystem::UserMemoryMap& ionBufferMap) const noexcept{
    return SNPE_SUCCESS == Snpe_PSNPE_RegisterIonBuffers(handle(), getHandle(ionBufferMap));
  }
  /* To be deprecated, please use new api deregisterMemoryMappedBuffers */
  bool deregisterIonBuffers(const DlSystem::StringList& ionBufferNames) const noexcept{
    return SNPE_SUCCESS == Snpe_PSNPE_DeregisterIonBuffers(handle(), getHandle(ionBufferNames));
  }

  bool registerMemoryMappedBuffers(const DlSystem::UserMemoryMap& memoryMappedBufferMap) noexcept{
    return SNPE_SUCCESS == Snpe_PSNPE_RegisterUserMemoryMappedBuffers(handle(), getHandle(memoryMappedBufferMap));
  }

  bool deregisterMemoryMappedBuffers(const DlSystem::StringList& bufferNames) noexcept{
    return SNPE_SUCCESS == Snpe_PSNPE_DeregisterUserMemoryMappedBuffers(handle(), getHandle(bufferNames));
  }

  const char* getLastErrorString(){
    return Snpe_PSNPE_GetLastErrorString(handle());
  }

private:
  PSNPE(const PSNPE&) = delete;
  PSNPE& operator=(const PSNPE&) = delete;

};

} // ns PSNPE



ALIAS_IN_ZDL_NAMESPACE(PSNPE, BuildMode)
ALIAS_IN_ZDL_NAMESPACE(PSNPE, InputOutputTransmissionMode)
ALIAS_IN_ZDL_NAMESPACE(PSNPE, OutputAsyncCallbackParam)
ALIAS_IN_ZDL_NAMESPACE(PSNPE, InputOutputAsyncCallbackParam)
ALIAS_IN_ZDL_NAMESPACE(PSNPE, InputOutputInputAsyncCallbackParam)

ALIAS_IN_ZDL_NAMESPACE(PSNPE, OutputAsyncCallbackFunc)
ALIAS_IN_ZDL_NAMESPACE(PSNPE, InputOutputAsyncCallbackFunc)
ALIAS_IN_ZDL_NAMESPACE(PSNPE, InputOutputAsyncInputCallback)
ALIAS_IN_ZDL_NAMESPACE(PSNPE, BuildConfig)
ALIAS_IN_ZDL_NAMESPACE(PSNPE, PSNPE)
