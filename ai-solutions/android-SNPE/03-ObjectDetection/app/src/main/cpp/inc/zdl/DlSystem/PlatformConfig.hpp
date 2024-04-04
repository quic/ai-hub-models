//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"

#include "DlSystem/PlatformConfig.h"

namespace DlSystem {

struct UserGLConfig
{
   /// Holds user EGL context.
   ///
   void* userGLContext = nullptr;

   /// Holds user EGL display.
   void* userGLDisplay = nullptr;
};

struct UserGpuConfig{
   /// Holds user OpenGL configuration.
   ///
   UserGLConfig userGLConfig;
};

class PlatformConfig : public Wrapper<PlatformConfig, Snpe_PlatformConfig_Handle_t> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_PlatformConfig_Delete};

  class UserGLConfigInternal : public Wrapper<UserGLConfigInternal, Snpe_UserGLConfig_Handle_t, true> {
    friend BaseType;
    // Use this to get free move Ctor and move assignment operator, provided this class does not specify
    // as copy assignment operator or copy Ctor
    using BaseType::BaseType;

    static constexpr DeleteFunctionType DeleteFunction{Snpe_UserGLConfig_Delete};

  public:
    UserGLConfigInternal()
      : BaseType(Snpe_UserGLConfig_Create())
    {  }
    UserGLConfigInternal(const UserGLConfig& uglc)
      : UserGLConfigInternal()
    {
      setUserGLContext(uglc.userGLContext);
      setUserGLDisplay(uglc.userGLDisplay);
    }
    void setUserGLContext(void* userGLContext){
      Snpe_UserGLConfig_SetUserGLContext(handle(), userGLContext);
    }
    void setUserGLDisplay(void* userGLDisplay){
      Snpe_UserGLConfig_SetUserGLDisplay(handle(), userGLDisplay);
    }

    void* getUserGLContext(){
      return Snpe_UserGLConfig_GetUserGLContext(handle());
    }
    void* getUserGLDisplay(){
      return Snpe_UserGLConfig_GetUserGLDisplay(handle());
    }
  };



  class UserGpuConfigInternal : public Wrapper<UserGpuConfigInternal, Snpe_UserGpuConfig_Handle_t, true> {
    friend BaseType;
    // Use this to get free move Ctor and move assignment operator, provided this class does not specify
    // as copy assignment operator or copy Ctor
    using BaseType::BaseType;

    static constexpr DeleteFunctionType DeleteFunction{Snpe_UserGpuConfig_Delete};

  public:
    UserGpuConfigInternal()
      : BaseType(Snpe_UserGpuConfig_Create())
    {  }

    void set(const UserGLConfig& userGLConfig){
      UserGLConfigInternal uglc(userGLConfig);
      Snpe_UserGpuConfig_Set(handle(), getHandle(uglc));
    }

    void get(UserGLConfig& uglc){
      UserGLConfigInternal uglci(moveHandle(Snpe_UserGpuConfig_Get_Ref(handle()), true));

      uglc.userGLContext = uglci.getUserGLContext();
      uglc.userGLDisplay = uglci.getUserGLDisplay();
    }

  };
public:

  /**
    * @brief .
    *
    * An enum class of all supported platform types
    */
  enum class PlatformType_t
  {
    /// Unknown platform type.
    UNKNOWN = 0,

    /// Snapdragon CPU.
    CPU = 1,

    /// Adreno GPU.
    GPU = 2,

    /// Hexagon DSP.
    DSP = 3
  };

  /**
    * @brief .
    *
    * A union class user platform configuration information
    */
  struct PlatformConfigInfo
  {
    /// Holds user GPU Configuration.
    ///
    UserGpuConfig userGpuConfig;

  };

  ~PlatformConfig() = default;

  PlatformConfig()
    : BaseType(Snpe_PlatformConfig_Create())
  {  }

  PlatformConfig(const PlatformConfig& other)
    : BaseType(Snpe_PlatformConfig_CreateCopy(other.handle()))
  {  }

  /**
    * @brief Retrieves the platform type
    *
    * @return Platform type
    */
  PlatformType_t getPlatformType() const{
    return static_cast<PlatformType_t>(Snpe_PlatformConfig_GetPlatformType(handle()));
  };

  /**
    * @brief Indicates whther the plaform configuration is valid.
    *
    * @return True if the platform configuration is valid; false otherwise.
    */
  bool isValid() const{
    return Snpe_PlatformConfig_IsValid(handle());
  };

  /**
    * @brief Retrieves the Gpu configuration
    *
    * @param[out] userGpuConfig The passed in userGpuConfig populated with the Gpu configuration on return.
    *
    * @return True if Gpu configuration was retrieved; false otherwise.
    */
  bool getUserGpuConfig(UserGpuConfig& userGpuConfig) const{
    auto platformType = static_cast<PlatformType_t>(Snpe_PlatformConfig_GetPlatformType(handle()));
    if(platformType != PlatformType_t::GPU) return false;

    UserGpuConfigInternal gpuConf(moveHandle(Snpe_PlatformConfig_GetUserGpuConfig(handle())));

    gpuConf.get(userGpuConfig.userGLConfig);
    return true;
  }

  /**
    * @brief Sets the Gpu configuration
    *
    * @param[in] userGpuConfig Gpu Configuration
    *
    * @return True if Gpu configuration was successfully set; false otherwise.
    */
  bool setUserGpuConfig(UserGpuConfig& userGpuConfig){
    UserGpuConfigInternal gpuConf;
    gpuConf.set(userGpuConfig.userGLConfig);
    return Snpe_PlatformConfig_SetUserGpuConfig(handle(), getHandle(gpuConf));
  }

  /**
    * @brief Sets the platform options
    *
    * @param[in] options Options as a string in the form of "keyword:options"
    *
    * @return True if options are pass validation; otherwise false.  If false, the options are not updated.
    */
  bool setPlatformOptions(const std::string& options){
    return Snpe_PlatformConfig_SetPlatformOptions(handle(), options.c_str());
  }

  /**
    * @brief Indicates whther the plaform configuration is valid.
    *
    * @return True if the platform configuration is valid; false otherwise.
    */
  bool isOptionsValid() const{
    return Snpe_PlatformConfig_IsOptionsValid(handle());
  }

  /**
    * @brief Gets the platform options
    *
    * @return Options as a string
    */
  std::string getPlatformOptions() const {
    return Snpe_PlatformConfig_GetPlatformOptions(handle());
  }

  /**
    * @brief Sets the platform options
    *
    * @param[in] optionName Name of platform options"
    * @param[in] value Value of specified optionName
    *
    * @return If true, add "optionName:value" to platform options if optionName don't exist, otherwise update the
    *         value of specified optionName.
    *         If false, the platform options will not be changed.
    */
  bool setPlatformOptionValue(const std::string& optionName, const std::string& value){
    return Snpe_PlatformConfig_SetPlatformOptionValue(handle(), optionName.c_str(), value.c_str());
  }

  /**
     * @brief Removes the platform options
     *
     * @param[in] optionName Name of platform options"
     * @param[in] value Value of specified optionName
     *
     * @return If true, removed "optionName:value" to platform options if optionName don't exist, do nothing.
     *         If false, the platform options will not be changed.
     */
  bool removePlatformOptionValue(const std::string& optionName, const std::string& value){
    return Snpe_PlatformConfig_RemovePlatformOptionValue(handle(), optionName.c_str(), value.c_str());
  }

  static void SetIsUserGLBuffer(bool isUserGLBuffer){
    Snpe_PlatformConfig_SetIsUserGLBuffer(isUserGLBuffer);
  }
  static bool GetIsUserGLBuffer(){
    return Snpe_PlatformConfig_GetIsUserGLBuffer();
  }

};


} // ns DlSystem


ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserGLConfig)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserGpuConfig)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, PlatformConfig)
