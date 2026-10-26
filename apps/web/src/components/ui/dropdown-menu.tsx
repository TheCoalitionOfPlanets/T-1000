import * as Dropdown from "@radix-ui/react-dropdown-menu"
import type { ComponentProps } from "react"
import { cn } from "@/lib/utils"

export const DropdownMenu = Dropdown.Root
export const DropdownMenuTrigger = Dropdown.Trigger
export function DropdownMenuContent({ className, sideOffset = 6, ...props }: ComponentProps<typeof Dropdown.Content>) {
  return <Dropdown.Portal><Dropdown.Content className={cn("dropdown-content", className)} sideOffset={sideOffset} {...props} /></Dropdown.Portal>
}
export function DropdownMenuItem({ className, ...props }: ComponentProps<typeof Dropdown.Item>) {
  return <Dropdown.Item className={cn("dropdown-item", className)} {...props} />
}
