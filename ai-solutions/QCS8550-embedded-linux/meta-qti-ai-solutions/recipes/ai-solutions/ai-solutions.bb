inherit cmake pkgconfig

HOMEPAGE         = "http://support.cdmatech.com"
LICENSE          = "Qualcomm-Technologies-Inc.-Proprietary"
LIC_FILES_CHKSUM = "file://${COREBASE}/meta-qti-bsp-prop/files/qcom-licenses/\
${LICENSE};md5=92b1d0ceea78229551577d4284669bb8"

SUMMARY = "AI-Solutions on QCS8550"
DESCRIPTION = "AI-Solutions"

LICENSE = "Qualcomm-Technologies-Inc.-Proprietary"

SRC_URI = "file://app"
S = "${WORKDIR}/app"

DEPENDS += " jsoncpp json-glib gflags gstreamer1.0 gstreamer1.0-plugins-base opencv  snpe"

do_install(){
    install -d ${D}/${bindir}
    install -m 0777 ${WORKDIR}/build/out/ai-solutions ${D}/${bindir}
}

INSANE_SKIP_${PN} += "arch"

FILES_${PN} += "${bindir}/*"
