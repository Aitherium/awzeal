/* Flat image for ZealOS: header first, everything (bss too) in one blob. */
OUTPUT_FORMAT(elf64-x86-64)
ENTRY(zl_header)
SECTIONS
{
    . = ZL_BASE;
    .image : {
        KEEP(*(.text.zhdr))
        *(.text .text.*)
        . = ALIGN(64);
        *(.rodata .rodata.*)
        . = ALIGN(64);
        *(.data .data.*)
        . = ALIGN(64);
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(64);
    }
    /DISCARD/ : { *(.note*) *(.comment) *(.eh_frame) }
}
