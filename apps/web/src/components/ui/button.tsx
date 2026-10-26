import { Slot } from "@radix-ui/react-slot"
import type { ComponentProps } from "react"
import { cn } from "@/lib/utils"

export function Button({ className, variant = "default", size = "default", asChild = false, ...props }: ComponentProps<"button"> & {
  variant?: "default" | "ghost" | "outline" | "secondary"
  size?: "default" | "icon" | "sm" | "lg"
  asChild?: boolean
}) {
  const Component = asChild ? Slot : "button"
  return <Component className={cn("ui-button", `ui-button-${variant}`, `ui-button-${size}`, className)} {...props} />
}
